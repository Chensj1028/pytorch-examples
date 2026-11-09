# holds various wrapping policies for fsdp
# 保存 FSDP 的各种包装策略。


import torch.distributed as dist
import torch.nn as nn
import torch

from transformers.models.t5.modeling_t5 import T5Block

from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullyShardedDataParallel as FSDP,
    CPUOffload,
    BackwardPrefetch,
    MixedPrecision,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    size_based_auto_wrap_policy,
    enable_wrap,
    wrap,
)

import functools
from typing import Type


def get_size_policy(min_params=1e8):
    num_wrap_policy = functools.partial(
        size_based_auto_wrap_policy, min_num_params=min_params
    )
    return num_wrap_policy


def get_t5_wrapper():
    """we register our main layer class and use the fsdp transformer wrapping policy
    ensures embedding layers are in the root fsdp unit for shared access and that fsdp units map to transformer layers
    """
    # 注册主要层的类，并使用 FSDP 的 Transformer 包装策略。
    # 确保嵌入层位于根 FSDP 单元中以便共享访问，同时使 FSDP 单元与 Transformer 层对应。
    # ====   use new transformer wrapper
    # 使用新的 Transformer 包装器。

    t5_auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            T5Block,
        },
    )

    return t5_auto_wrap_policy
