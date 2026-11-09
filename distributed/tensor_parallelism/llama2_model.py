# Copyright (c) Meta Platforms, Inc. and affiliates.
# 版权所有 © Meta Platforms, Inc. 及其关联方。
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.
# 本软件可依照 Llama 2 社区许可协议的条款使用和分发。

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1  # defined later by tokenizer
    # 之后由分词器定义。
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    # 使 SwiGLU 隐藏层大小成为较大 2 的幂的整数倍。
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5

    max_batch_size: int = 32
    max_seq_len: int = 32768
    # If `True`, then each transformer block init uses its layer ID, and if
    # 若为 True，每个 Transformer 块使用自身层编号初始化；
    # `False`, each uses the total number of transformer blocks
    # 若为 False，则每个块都使用 Transformer 块总数。
    depth_init: bool = True


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim'
    and the end index 'end'. The 'theta' parameter scales the frequencies.
    The returned tensor contains complex values in complex64 data type.

    Args:
        dim (int): Dimension of the frequency tensor.
        end (int): End index for precomputing frequencies.
        theta (float, optional): Scaling factor for frequency computation. Defaults to 10000.0.

    Returns:
        torch.Tensor: Precomputed frequency tensor with complex exponentials.
    """
    # 预计算给定维度的复指数频率张量。
    # 使用维度 dim 和结束位置 end 计算频率，theta 控制频率缩放；返回 complex64 张量。
    # 参数 dim（int）：频率张量维度。end（int）：预计算的结束索引。
    # 参数 theta（float，可选）：频率缩放因子，默认 10000.0。
    # 返回值 torch.Tensor：预计算的复指数频率张量。
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    Reshape frequency tensor for broadcasting it with another tensor.

    This function reshapes the frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.

    Returns:
        torch.Tensor: Reshaped frequency tensor.
    """
    # 调整频率张量形状，以便与目标张量进行广播。
    # 参数 freqs_cis（torch.Tensor）：待调整的频率张量。
    # 参数 x（torch.Tensor）：要求广播兼容的目标张量。
    # 返回值：调整形状后的频率张量，用于逐元素运算。
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query 'xq' and key 'xk' tensors using the provided
    frequency tensor 'freqs_cis'. The input tensors are reshaped as complex numbers, and the frequency tensor
    is reshaped for broadcasting compatibility. The resulting tensors contain rotary embeddings and are
    returned as real tensors.

    Args:
        xq (torch.Tensor): Query tensor to apply rotary embeddings.
        xk (torch.Tensor): Key tensor to apply rotary embeddings.
        freqs_cis (torch.Tensor): Precomputed frequency tensor for complex exponentials.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
    """
    # 使用频率张量为查询 xq 和键 xk 应用旋转位置嵌入。
    # 将输入重排为复数，并调整 freqs_cis 的形状以满足广播要求，最后返回实数张量。
    # 参数 xq、xk：查询张量和键张量。freqs_cis：预计算的复指数频率张量。
    # 返回值：应用旋转位置嵌入后的查询张量和键张量组成的二元组。
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class RMSNorm(nn.Module):
    """
    Initialize the RMSNorm normalization layer.

    Args:
        dim (int): The dimension of the input tensor.
        eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

    Attributes:
        eps (float): A small value added to the denominator for numerical stability.
        weight (nn.Parameter): Learnable scaling parameter.

    """
    # 初始化 RMSNorm 归一化层。
    # 参数 dim（int）：输入维度。eps（float，可选）：分母中的稳定性常数，默认 1e-6。
    # 属性 eps：数值稳定性常数。weight（nn.Parameter）：可学习的缩放参数。

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)  # type: ignore


class Attention(nn.Module):
    """
    Multi-head attention module.

    Args:
        model_args (ModelArgs): Model configuration arguments.

    Attributes:
        n_kv_heads (int): Number of key and value heads.
        n_heads (int): Number of query heads.
        n_local_kv_heads (int): Number of local key and value heads.
        n_rep (int): Number of repetitions for local heads.
        head_dim (int): Dimension size of each attention head.
        wq (Linear): Linear transformation for queries.
        wk (Linear): Linear transformation for keys.
        wv (Linear): Linear transformation for values.
        wo (Linear): Linear transformation for output.

    """
    # 多头注意力模块。
    # 参数 model_args（ModelArgs）：模型配置。
    # 属性 n_kv_heads：键和值的头数；n_heads：查询头数；n_local_kv_heads：本地键和值的头数。
    # n_rep：本地头的重复次数；head_dim：每个注意力头的维度。
    # wq、wk、wv、wo：分别用于查询、键、值和输出的线性变换。

    def __init__(self, model_args: ModelArgs):
        super().__init__()
        self.n_heads = model_args.n_heads
        self.n_kv_heads = (
            model_args.n_heads
            if model_args.n_kv_heads is None
            else model_args.n_kv_heads
        )
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = model_args.dim // model_args.n_heads

        self.wq = nn.Linear(
            model_args.dim, model_args.n_heads * self.head_dim, bias=False
        )
        self.wk = nn.Linear(model_args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(model_args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(
            model_args.n_heads * self.head_dim, model_args.dim, bias=False
        )

    def init_weights(self, init_std: float):
        for linear in (self.wq, self.wk, self.wv):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.wo.weight, mean=0.0, std=init_std)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
    ):
        """
        Forward pass of the attention module.

        Args:
            x (torch.Tensor): Input tensor.
            freqs_cis (torch.Tensor): Precomputed frequency tensor.

        Returns:
            torch.Tensor: Output tensor after attention.

        """
        # 注意力模块的前向传播。
        # 参数 x：输入张量；freqs_cis：预计算的频率张量。
        # 返回值：经过注意力计算后的输出张量。
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        keys = repeat_kv(xk, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)
        # 形状：批大小、序列长度、本地注意力头数、每个头的维度。
        values = repeat_kv(xv, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)
        # 形状：批大小、序列长度、本地注意力头数、每个头的维度。

        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        # 形状：批大小、本地注意力头数、序列长度、每个头的维度。
        xk = keys.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        # 形状：批大小、本地注意力头数、序列长度、每个头的维度。
        xv = values.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        # 形状：批大小、本地注意力头数、序列长度、每个头的维度。

        # we use casual mask for training
        # 训练时使用因果掩码。
        output = F.scaled_dot_product_attention(xq, xk, xv, is_causal=True)
        output = output.transpose(
            1, 2
        ).contiguous()  # (bs, seqlen, n_local_heads, head_dim)
        # 形状：批大小、序列长度、本地注意力头数、每个头的维度。
        output = output.view(bsz, seqlen, -1)
        return self.wo(output)


class FeedForward(nn.Module):
    """
    FeedForward module

    Args:
        dim (int): Input dimension.
        hidden_dim (int): Hidden dimension of the feedforward layer.
        multiple_of (int): Value to ensure hidden dimension is a multiple of this value.
        ffn_dim_multiplier (Optional[float]): Custom multiplier for hidden dimension. Defaults to None.

    Attributes:
        w1 (Linear): Linear transformation for the first layer.
        w2 (Linear): Linear transformation for the second layer.
        w3 (Linear): Linear transformation for the third layer.

    """
    # 前馈网络模块。
    # 参数 dim：输入维度；hidden_dim：前馈层的隐藏维度。
    # multiple_of：保证隐藏维度是该值的整数倍；ffn_dim_multiplier：自定义隐藏维度倍数，默认 None。
    # 属性 w1、w2、w3：第一、第二和第三个线性变换层。

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        # 自定义维度缩放因子。
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

    def init_weights(self, init_std: float):
        nn.init.trunc_normal_(self.w1.weight, mean=0.0, std=0.02)
        for linear in (self.w2, self.w3):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=init_std)


class TransformerBlock(nn.Module):
    """
    TransformerBlock Module

    Args:
        layer_id (int): Identifier for the layer.
        model_args (ModelArgs): Model configuration arguments.

    Attributes:
        n_heads (int): Number of attention heads.
        dim (int): Dimension size of the model.
        head_dim (int): Dimension size of each attention head.
        attention (Attention): Attention module.
        feed_forward (FeedForward): FeedForward module.
        layer_id (int): Identifier for the layer.
        attention_norm (RMSNorm): Layer normalization for attention output.
        ffn_norm (RMSNorm): Layer normalization for feedforward output.

    """
    # Transformer 块模块。
    # 参数 layer_id：层编号；model_args：模型配置。
    # 属性 n_heads：注意力头数；dim：模型维度；head_dim：每个头的维度。
    # attention：注意力模块；feed_forward：前馈模块；layer_id：层编号。
    # attention_norm、ffn_norm：分别用于注意力输出和前馈输出的 RMSNorm 归一化层。

    def __init__(self, layer_id: int, model_args: ModelArgs):
        super().__init__()
        self.n_heads = model_args.n_heads
        self.dim = model_args.dim
        self.attention = Attention(model_args)
        self.feed_forward = FeedForward(
            dim=model_args.dim,
            hidden_dim=4 * model_args.dim,
            multiple_of=model_args.multiple_of,
            ffn_dim_multiplier=model_args.ffn_dim_multiplier,
        )
        self.layer_id = layer_id
        self.num_layers = model_args.n_layers

        self.attention_norm = RMSNorm(
            dim=model_args.dim, eps=model_args.norm_eps
        )
        self.ffn_norm = RMSNorm(
            dim=model_args.dim, eps=model_args.norm_eps
        )

        if model_args.depth_init:
            self.weight_init_std = 0.02 / (2 * (self.layer_id + 1)) ** 0.5
        else:
            self.weight_init_std = 0.02 / (2 * self.num_layers) ** 0.5

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
    ):
        """
        Perform a forward pass through the TransformerBlock.

        Args:
            x (torch.Tensor): Input tensor.
            freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies.

        Returns:
            torch.Tensor: Output tensor after applying attention and feedforward layers.

        """
        # 执行 Transformer 块的前向传播。
        # 参数 x：输入张量；freqs_cis：预计算的余弦和正弦频率。
        # 返回值：经过注意力层和前馈层后的输出张量。
        h = x + self.attention(self.attention_norm(x), freqs_cis)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

    def init_weights(self):
        for norm in (self.attention_norm, self.ffn_norm):
            norm.reset_parameters()
        self.attention.init_weights(self.weight_init_std)
        self.feed_forward.init_weights(self.weight_init_std)


class Transformer(nn.Module):
    """
    Transformer Module

    Args:
        model_args (ModelArgs): Model configuration arguments.

    Attributes:
        model_args (ModelArgs): Model configuration arguments.
        vocab_size (int): Vocabulary size.
        n_layers (int): Number of layers in the model.
        tok_embeddings (ParallelEmbedding): Token embeddings.
        layers (torch.nn.ModuleList): List of Transformer blocks.
        norm (RMSNorm): Layer normalization for the model output.
        output (ColumnParallelLinear): Linear layer for final output.
        freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies.

    """
    # Transformer 模块。
    # 参数及属性 model_args：模型配置。
    # 属性 vocab_size：词表大小；n_layers：模型层数；tok_embeddings：词元嵌入。
    # layers：Transformer 块列表；norm：模型输出的 RMSNorm；output：最终输出线性层。
    # freqs_cis：预计算的余弦和正弦频率张量。

    def __init__(self, model_args: ModelArgs):
        super().__init__()
        self.model_args = model_args
        self.vocab_size = model_args.vocab_size
        self.n_layers = model_args.n_layers
        self.model_dim = model_args.dim

        self.tok_embeddings = nn.Embedding(model_args.vocab_size, model_args.dim)
        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(
                model_args.dim // model_args.n_heads,
                # Need to compute until at least the max token limit for generation
                # 至少计算到生成时所需的最大词元数量，
                # (use 2x max sequence length to be safe)
                # 为稳妥起见，使用最大序列长度的两倍。
                model_args.max_seq_len * 2,
            ),
        )
        self.layers = torch.nn.ModuleList()
        for layer_id in range(model_args.n_layers):
            self.layers.append(TransformerBlock(layer_id, model_args))

        self.norm = RMSNorm(
            dim=model_args.dim, eps=model_args.norm_eps
        )

        self.output = nn.Linear(model_args.dim, model_args.vocab_size, bias=False)
        self.init_weights()

    def init_weights(self):
        """
        [Note: On ``init_weights`` vs. ``reset_parameters``]
        Modules may define ``reset_parameters`` to initialize parameter values.
        ``reset_parameters`` is meant to only initialize directly owned
        parameters/buffers, not those of their child modules, and it can be
        used to give the initial values for these tensors.
        Separately, users may want custom initialization for their modules,
        different from that in ``reset_parameters``. For this, we define
        ``init_weights``. We only call it in the constructor of this
        ``Transformer`` root module to avoid reinitializing tensors.
        """
        # 关于 init_weights 与 reset_parameters 的区别：
        # 模块可定义 reset_parameters，初始化其直接拥有的参数和缓冲区，而不初始化子模块。
        # 用户也可能需要不同于 reset_parameters 的自定义初始化，因此定义 init_weights。
        # 这里只在根 Transformer 模块的构造函数中调用它，避免重复初始化张量。
        with torch.device(self.freqs_cis.device):
            self.freqs_cis = precompute_freqs_cis(
                self.model_args.dim // self.model_args.n_heads,
                # Need to compute until at least the max token limit for generation
                # 至少计算到生成时所需的最大词元数量，
                # (use 2x max sequence length to be safe)
                # 为稳妥起见，使用最大序列长度的两倍。
                self.model_args.max_seq_len * 2,
            )
        nn.init.normal_(self.tok_embeddings.weight)
        for layer in self.layers:
            layer.init_weights()
        self.norm.reset_parameters()
        final_out_std = self.model_args.dim**-0.5
        cutoff_factor = 3
        nn.init.trunc_normal_(
            self.output.weight,
            mean=0.0,
            std=final_out_std,
            a=-cutoff_factor * final_out_std,
            b=cutoff_factor * final_out_std,
        )

    def forward(self, tokens: torch.Tensor):
        """
        Perform a forward pass through the Transformer model.

        Args:
            tokens (torch.Tensor): Input token indices.

        Returns:
            torch.Tensor: Output logits after applying the Transformer model.

        """
        # 执行 Transformer 模型的前向传播。
        # 参数 tokens：输入词元索引。
        # 返回值：经过 Transformer 后的输出 logits。
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[0:seqlen]

        for layer in self.layers:
            h = layer(h, freqs_cis)
        h = self.norm(h)
        output = self.output(h).float()
        return output

    @classmethod
    def from_model_args(cls, model_args: ModelArgs) -> "Transformer":
        """
        Initialize a Transformer model from a ModelArgs object.

        Args:
            model_args (ModelArgs): Model configuration arguments.

        Returns:
            Transformer: Transformer model.

        """
        # 根据 ModelArgs 对象初始化 Transformer。
        # 参数 model_args：模型配置。
        # 返回值：Transformer 模型。
        return cls(model_args)
