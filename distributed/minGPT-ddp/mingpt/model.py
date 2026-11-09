"""
Full definition of a GPT Language Model, all of it in this single file.
Adapted from https://github.com/karpathy/minGPT/blob/master/mingpt/model.py
"""
# 本文件包含 GPT 语言模型的完整定义。
# 改编自上方所列的 minGPT 源码。

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    model_type: str = 'gpt2'
    # model configurations
    # 模型配置。
    n_layer: int = None
    n_head: int = None
    n_embd: int =  None
    # openai's values for gpt2
    # OpenAI 为 GPT-2 使用的配置值。
    vocab_size: int = 50257 
    block_size: int = 1024
    # dropout hyperparameters
    # Dropout 超参数。
    embd_pdrop: float = 0.1
    resid_pdrop: float = 0.1
    attn_pdrop: float = 0.1

@dataclass
class OptimizerConfig:
    learning_rate: float = 3e-4
    weight_decay: float = 0.1

class MultiheadAttentionLayer(nn.Module):
    """
    A multi-head masked self-attention layer with a projection at the end.
    """
    # 带掩码的多头自注意力层，末尾包含投影层。

    def __init__(self, config, device="cpu", dtype=torch.float32):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, device=device, dtype=dtype)
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))
        self.attn = torch.nn.MultiheadAttention(
            embed_dim=config.n_embd,
            num_heads=config.n_head,
            dropout=config.attn_pdrop,
            batch_first=True,
            device=device,
            dtype=dtype
        )

    def forward(self, x):
        _, seq_size, _ = x.size()
        y = self.attn(x, x, x, attn_mask=self.mask[0, 0, :seq_size, :seq_size])[0]
        y = self.resid_drop(self.c_proj(y))
        return y

class Block(nn.Module):
    """ an unassuming Transformer block """
    # 一个简单的 Transformer 块。
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = MultiheadAttentionLayer(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class EmbeddingStem(nn.Module):
    def __init__(self, config: GPTConfig, device="cpu", dtype=torch.float32):
        super().__init__()
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd, device=device, dtype=dtype)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd, device=device, dtype=dtype))
        self.drop = nn.Dropout(config.embd_pdrop)
        self.block_size = config.block_size

    def reset_parameters(self):
        self.tok_emb.reset_parameters()

    def forward(self, idx):
        b, t = idx.size()
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"

        token_embeddings = self.tok_emb(idx)  # each index maps to a (learnable) embedding vector
        # 每个索引映射到一个可学习的嵌入向量。
        position_embeddings = self.pos_emb[:, :t, :]  # each position maps to a (learnable) position vector
        # 每个位置映射到一个可学习的位置向量。
        return self.drop(token_embeddings + position_embeddings)
        
class GPT(nn.Module):
    """ GPT Language Model """
    # GPT 语言模型。

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.block_size = config.block_size
        config = self._set_model_config(config)

        # input embedding stem
        # 输入嵌入模块。
        self.emb_stem = EmbeddingStem(config)
        # transformer
        # Transformer 模块。
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        # decoder head
        # 解码输出头。
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # init all weights, and apply a special scaled init to the residual projections, per GPT-2 paper
        # 初始化所有权重，并按 GPT-2 论文对残差投影使用特殊的缩放初始化。
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                p.data.normal_(mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters (note we don't count the decoder parameters in lm_head)
        # 报告参数数量；注意，此处不计入 lm_head 中的解码器参数。
        n_params = sum(p.numel() for p in self.blocks.parameters())
        print("number of parameters: %.2fM" % (n_params/1e6,))

    def _set_model_config(self, config):
        type_given = config.model_type is not None
        params_given = all([config.n_layer is not None, config.n_head is not None, config.n_embd is not None])
        # assert type_given ^ params_given # exactly one of these (XOR)
        # 该断言要求两种配置方式恰好提供一种（异或）。
        if type_given and not params_given:
            # translate from model_type to detailed configuration
            # 将 model_type 转换为详细配置。
            config.__dict__.update({
                # names follow the huggingface naming conventions
                # 名称遵循 Hugging Face 的命名约定。
                # GPT-1
                # GPT 第 1 代模型。
                'openai-gpt':   dict(n_layer=12, n_head=12, n_embd=768),  # 117M params
                # 约 1.17 亿个参数。
                # GPT-2 configs
                # GPT-2 配置。
                'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
                # 约 1.24 亿个参数。
                'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
                # 约 3.5 亿个参数。
                'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
                # 约 7.74 亿个参数。
                'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
                # 约 15.58 亿个参数。
                # Gophers
                # Gopher 系列模型。
                'gopher-44m':   dict(n_layer=8, n_head=16, n_embd=512),
                # (there are a number more...)
                # 还有其他一些配置。
                # I made these tiny models up
                # 这些微型模型的配置由原作者自行设计。
                'gpt-mini':     dict(n_layer=6, n_head=6, n_embd=192),
                'gpt-micro':    dict(n_layer=4, n_head=4, n_embd=128),
                'gpt-nano':     dict(n_layer=3, n_head=3, n_embd=48),
            }[config.model_type])
        return config
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, idx, targets=None):
        x = self.emb_stem(idx)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)

        # if we are given some desired targets also calculate the loss
        # 若给出了目标值，则同时计算损失。
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, do_sample=False, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        # 给定条件索引序列 idx（形状为 (b, t) 的 LongTensor），继续生成 max_new_tokens 个词元，
        # 每次将预测结果重新输入模型。通常应先切换到 model.eval() 评估模式。
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            # 如果上下文序列过长，就裁剪到 block_size。
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            # forward the model to get the logits for the index in the sequence
            # 执行模型前向传播，获得序列位置对应的 logits。
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            # 取最后一个时间步的 logits，并按指定温度缩放。
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            # 可选：仅保留得分最高的 k 个选项。
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            # 使用 softmax 将 logits 转换为归一化概率。
            probs = F.softmax(logits, dim=-1)
            # either sample from the distribution or take the most likely element
            # 从分布中采样，或选择概率最大的元素。
            if do_sample:
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                _, idx_next = torch.topk(probs, k=1, dim=-1)
            # append sampled index to the running sequence and continue
            # 将采样得到的索引追加到当前序列，继续生成。
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


def create_optimizer(model: torch.nn.Module, opt_config: OptimizerConfig):
    """
    This long function is unfortunately doing something very simple and is being very defensive:
    We are separating out all parameters of the model into two buckets: those that will experience
    weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
    We are then returning the PyTorch optimizer object.
    """
    # 这个较长的函数执行的操作很简单，但包含较多防御性检查：
    # 将模型参数分为两组，一组应用权重衰减正则化，另一组不应用；
    # 偏置、LayerNorm 权重和嵌入权重属于后一组。
    # 最后返回 PyTorch 优化器对象。

    # separate out all parameters to those that will and won't experience regularizing weight decay
    # 将参数分为需要和不需要权重衰减正则化的两组。
    decay = set()
    no_decay = set()
    whitelist_weight_modules = (torch.nn.Linear, )
    blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
            # 完整参数名。
            # random note: because named_modules and named_parameters are recursive
            # 补充说明：named_modules 和 named_parameters 都会递归遍历，
            # we will see the same tensors p many many times. but doing it this way
            # 因此同一个张量 p 会被多次遇到；采用这种方式
            # allows us to know which parent module any tensor p belongs to...
            # 可以确定每个张量 p 所属的父模块。
            if pn.endswith('bias'):
                # all biases will not be decayed
                # 所有偏置均不应用权重衰减。
                no_decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                # weights of whitelist modules will be weight decayed
                # 白名单模块的权重应用权重衰减。
                decay.add(fpn)
            elif pn.endswith('in_proj_weight'):
                # MHA projection layer
                # 多头注意力的投影层。
                decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                # weights of blacklist modules will NOT be weight decayed
                # 黑名单模块的权重不应用权重衰减。
                no_decay.add(fpn)
            elif pn.endswith('pos_emb'):
                # positional embedding shouldn't be decayed
                # 位置嵌入不应进行权重衰减。
                no_decay.add(fpn)

    # validate that we considered every parameter
    # 检查是否已覆盖每个参数。
    param_dict = {pn: p for pn, p in model.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
    assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                % (str(param_dict.keys() - union_params), )

    # create the pytorch optimizer object
    # 创建 PyTorch 优化器对象。
    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": opt_config.weight_decay},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=opt_config.learning_rate, betas=(0.9, 0.95))
    return optimizer