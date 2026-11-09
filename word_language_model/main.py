# coding: utf-8
import argparse
import time
import math
import os
import torch
import torch.nn as nn
import torch.onnx

import data
from model import PositionalEncoding, RNNModel, TransformerModel

parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM/GRU/Transformer Language Model')
parser.add_argument('--data', type=str, default='./data/wikitext-2',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of network (RNN_TANH, RNN_RELU, LSTM, GRU, Transformer)')
parser.add_argument('--emsize', type=int, default=200,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=200,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=20,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default='model.pt',
                    help='path to save the final model')
parser.add_argument('--onnx-export', type=str, default='',
                    help='path to export the final model in onnx format')
parser.add_argument('--nhead', type=int, default=2,
                    help='the number of heads in the encoder/decoder of the transformer model')
parser.add_argument('--dry-run', action='store_true',
                    help='verify the code and the model')
parser.add_argument('--accel', action='store_true',
                    help='Enables accelerated training')
parser.add_argument('--use-optimizer', action='store_true',
                    help='Uses AdamW optimizer for gradient updating')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
# 手动设置随机种子，以便复现。
torch.manual_seed(args.seed)

if args.accel and torch.accelerator.is_available():
    device = torch.accelerator.current_accelerator()

else:
    device = torch.device("cpu")

print("Using device:", device)

###############################################################################
# Load data
# 加载数据。
###############################################################################

corpus = data.Corpus(args.data)

# Starting from sequential data, batchify arranges the dataset into columns.
# batchify 从连续序列开始，将数据集整理成多列。
# For instance, with the alphabet as the sequence and batch size 4, we'd get
# 例如，输入为字母表序列、批大小为 4 时，将得到
# ┌ a g m s ┐
# │ b h n t │
# │ c i o u │
# │ d j p v │
# │ e k q w │
# └ f l r x ┘.
# These columns are treated as independent by the model, which means that the
# 模型将各列视为独立序列，因此
# dependence of e. g. 'g' on 'f' can not be learned, but allows more efficient
# 无法学习跨列的依赖，例如 g 对 f 的依赖，但这样可以更高效地
# batch processing.
# 进行批处理。

def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    # 计算数据集可以完整划分成多少个含 bsz 个元素的分组。
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    # 去掉无法完整分组的多余元素，即余数部分。
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    # 将数据均匀分成 bsz 列，对应各批次位置。
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)

eval_batch_size = 10
train_data = batchify(corpus.train, args.batch_size)
val_data = batchify(corpus.valid, eval_batch_size)
test_data = batchify(corpus.test, eval_batch_size)

###############################################################################
# Build the model
# 构建模型。
###############################################################################

ntokens = len(corpus.dictionary)
if args.model == 'Transformer':
    model = TransformerModel(ntokens, args.emsize, args.nhead, args.nhid, args.nlayers, args.dropout).to(device)
else:
    model = RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied).to(device)

criterion = nn.NLLLoss()
if args.use_optimizer:
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

###############################################################################
# Training code
# 训练代码。
###############################################################################

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    # 用新张量包装隐藏状态，使其与此前的计算历史分离。

    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


# get_batch subdivides the source data into chunks of length args.bptt.
# get_batch 将源数据切成长度为 args.bptt 的片段。
# If source is equal to the example output of the batchify function, with
# 若源数据与 batchify 函数示例的输出相同，且
# a bptt-limit of 2, we'd get the following two Variables for i = 0:
# bptt 上限为 2，则在 i = 0 时得到下面两个变量：
# ┌ a g m s ┐ ┌ b h n t ┐
# └ b h n t ┘ └ c i o u ┘
# Note that despite the name of the function, the subdivison of data is not
# 注意，虽然函数名称中有 batch，但此处的切分
# done along the batch dimension (i.e. dimension 1), since that was handled
# 不是沿批次维度，即第 1 维进行；那一步已经由
# by the batchify function. The chunks are along dimension 0, corresponding
# batchify 完成。这里沿第 0 维切分，
# to the seq_len dimension in the LSTM.
# 对应 LSTM 的序列长度维度。

def get_batch(source, i):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target


def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    # 启用评估模式，关闭 Dropout。
    model.eval()
    total_loss = 0.
    ntokens = len(corpus.dictionary)
    if args.model != 'Transformer':
        hidden = model.init_hidden(eval_batch_size)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, args.bptt):
            data, targets = get_batch(data_source, i)
            if args.model == 'Transformer':
                output = model(data)
                output = output.view(-1, ntokens)
            else:
                output, hidden = model(data, hidden)
                hidden = repackage_hidden(hidden)
            total_loss += len(data) * criterion(output, targets).item()
    return total_loss / (len(data_source) - 1)


def train():
    # Turn on training mode which enables dropout.
    # 启用训练模式，开启 Dropout。
    model.train()
    total_loss = 0.
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    if args.model != 'Transformer':
        hidden = model.init_hidden(args.batch_size)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
        data, targets = get_batch(train_data, i)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # 每个批次开始时，将隐藏状态与此前产生它的计算历史分离。
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        # 否则，模型会尝试一直反向传播到数据集的起点。
        if args.use_optimizer:
            optimizer.zero_grad()
        else:
            model.zero_grad()
        if args.model == 'Transformer':
            output = model(data)
            output = output.view(-1, ntokens)
        else:
            hidden = repackage_hidden(hidden)
            output, hidden = model(data, hidden)
        loss = criterion(output, targets)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        # clip_grad_norm 有助于防止 RNN/LSTM 中的梯度爆炸。
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        if args.use_optimizer:
            optimizer.step()
        else:
            for p in model.parameters():
                p.data.add_(p.grad, alpha=-lr)

        total_loss += loss.item()

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // args.bptt, lr,
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()
        if args.dry_run:
            break


def export_onnx(path, batch_size, seq_len):
    print('The model is also exported in ONNX format at {}.'.format(os.path.realpath(args.onnx_export)))
    model.eval()
    dummy_input = torch.LongTensor(seq_len * batch_size).zero_().view(-1, batch_size).to(device)
    hidden = model.init_hidden(batch_size)
    torch.onnx.export(model, (dummy_input, hidden), path)


# Loop over epochs.
# 按轮次循环。
lr = args.lr
best_val_loss = None

# At any point you can hit Ctrl + C to break out of training early.
# 随时按 Ctrl+C 可以提前结束训练。
try:
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train()
        val_loss = evaluate(val_data)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                           val_loss, math.exp(val_loss)))
        print('-' * 89)
        # Save the model if the validation loss is the best we've seen so far.
        # 若验证损失为目前最佳，则保存模型。
        if not best_val_loss or val_loss < best_val_loss:
            with open(args.save, 'wb') as f:
                torch.save(model, f)
            best_val_loss = val_loss
        else:
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            # 若验证集表现没有改善，则降低学习率。
            lr /= 4.0
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
# 加载保存的最佳模型。
with open(args.save, 'rb') as f:
    if args.model == 'Transformer':
        safe_globals = [
            PositionalEncoding,
            TransformerModel,
            torch.nn.functional.relu,
            torch.nn.modules.activation.MultiheadAttention,
            torch.nn.modules.container.ModuleList,
            torch.nn.modules.dropout.Dropout,
            torch.nn.modules.linear.Linear,
            torch.nn.modules.linear.NonDynamicallyQuantizableLinear,
            torch.nn.modules.normalization.LayerNorm,
            torch.nn.modules.sparse.Embedding,
            torch.nn.modules.transformer.TransformerEncoder,
            torch.nn.modules.transformer.TransformerEncoderLayer,
        ]
    else:
        safe_globals = [
            RNNModel,
            torch.nn.modules.dropout.Dropout,
            torch.nn.modules.linear.Linear,
            torch.nn.modules.rnn.GRU,
            torch.nn.modules.rnn.LSTM,
            torch.nn.modules.rnn.RNN,
            torch.nn.modules.sparse.Embedding,
        ]
    with torch.serialization.safe_globals(safe_globals):
        model = torch.load(f)
    # after load the rnn params are not a continuous chunk of memory
    # 加载后，RNN 参数可能不位于连续的内存块中，
    # this makes them a continuous chunk, and will speed up forward pass
    # 此操作将参数整理到连续内存，以加快前向传播。
    # Currently, only rnn model supports flatten_parameters function.
    # 目前仅 RNN 模型支持 flatten_parameters 函数。
    if args.model in ['RNN_TANH', 'RNN_RELU', 'LSTM', 'GRU']:
        model.rnn.flatten_parameters()

# Run on test data.
# 在测试数据上运行。
test_loss = evaluate(test_data)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)

if len(args.onnx_export) > 0:
    # Export the model in ONNX format.
    # 将模型导出为 ONNX 格式。
    export_onnx(args.onnx_export, batch_size=1, seq_len=args.bptt)
