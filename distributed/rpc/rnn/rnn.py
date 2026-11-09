import torch
import torch.nn as nn
import torch.distributed.rpc as rpc
from torch.distributed.rpc import RRef


def _call_method(method, rref, *args, **kwargs):
    r"""
    a helper function to call a method on the given RRef
    """
    # 在指定 RRef 上调用方法的辅助函数。
    return method(rref.local_value(), *args, **kwargs)


def _remote_method(method, rref, *args, **kwargs):
    r"""
    a helper function to run method on the owner of rref and fetch back the
    result using RPC
    """
    # 通过 RPC 在 RRef 所属节点运行方法，并取回结果的辅助函数。
    return rpc.rpc_sync(
        rref.owner(),
        _call_method,
        args=[method, rref] + list(args),
        kwargs=kwargs
    )


def _parameter_rrefs(module):
    r"""
    Create one RRef for each parameter in the given local module, and return a
    list of RRefs.
    """
    # 为给定本地模块的每个参数创建一个 RRef，并返回 RRef 列表。
    param_rrefs = []
    for param in module.parameters():
        param_rrefs.append(RRef(param))
    return param_rrefs


class EmbeddingTable(nn.Module):
    r"""
    Encoding layers of the RNNModel
    """
    # RNNModel 的编码层。
    def __init__(self, ntoken, ninp, dropout):
        super(EmbeddingTable, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        if torch.accelerator.is_available():
            device = torch.accelerator.current_accelerator()
            self.encoder = self.encoder.to(device)
        nn.init.uniform_(self.encoder.weight, -0.1, 0.1)

    def forward(self, input):
        if torch.accelerator.is_available():
            device = torch.accelerator.current_accelerator()
            input = input.to(device)
        return self.drop(self.encoder(input)).cpu()


class Decoder(nn.Module):
    r"""
    Decoding layers of the RNNModel
    """
    # RNNModel 的解码层。
    def __init__(self, ntoken, nhid, dropout):
        super(Decoder, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.decoder = nn.Linear(nhid, ntoken)
        nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder.weight, -0.1, 0.1)

    def forward(self, output):
        return self.decoder(self.drop(output))


class RNNModel(nn.Module):
    r"""
    A distributed RNN model which puts embedding table and decoder parameters on
    a remote parameter server, and locally holds parameters for the LSTM module.
    The structure of the RNN model is borrowed from the word language model
    example. See https://github.com/pytorch/examples/blob/main/word_language_model/model.py
    """
    # 分布式 RNN：嵌入表和解码器参数放在远程参数服务器，LSTM 参数保存在本地。
    # RNN 的结构借用词级语言模型示例，源码见上方链接。
    def __init__(self, ps, ntoken, ninp, nhid, nlayers, dropout=0.5):
        super(RNNModel, self).__init__()

        # setup embedding table remotely
        # 在远程设置嵌入表。
        self.emb_table_rref = rpc.remote(ps, EmbeddingTable, args=(ntoken, ninp, dropout))
        # setup LSTM locally
        # 在本地设置 LSTM。
        self.rnn = nn.LSTM(ninp, nhid, nlayers, dropout=dropout)
        # setup decoder remotely
        # 在远程设置解码器。
        self.decoder_rref = rpc.remote(ps, Decoder, args=(ntoken, nhid, dropout))

    def forward(self, input, hidden):
        # pass input to the remote embedding table and fetch emb tensor back
        # 将输入传给远程嵌入表，并取回嵌入张量。
        emb = _remote_method(EmbeddingTable.forward, self.emb_table_rref, input)
        output, hidden = self.rnn(emb, hidden)
        # pass output to the remote decoder and get the decoded output back
        # 将输出传给远程解码器，并取回解码结果。
        decoded = _remote_method(Decoder.forward, self.decoder_rref, output)
        return decoded, hidden

    def parameter_rrefs(self):
        remote_params = []
        # get RRefs of embedding table
        # 获取嵌入表的 RRef。
        remote_params.extend(_remote_method(_parameter_rrefs, self.emb_table_rref))
        # create RRefs for local parameters
        # 为本地参数创建 RRef。
        remote_params.extend(_parameter_rrefs(self.rnn))
        # get RRefs of decoder
        # 获取解码器的 RRef。
        remote_params.extend(_remote_method(_parameter_rrefs, self.decoder_rref))
        return remote_params
