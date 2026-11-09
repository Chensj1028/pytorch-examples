import torch

from torch.distributed.fsdp import (
    # FullyShardedDataParallel as FSDP,
    # CPUOffload,
    MixedPrecision,
    # BackwardPrefetch,
    # ShardingStrategy,
)

# requires grad scaler in main loop
# 主循环中需要使用梯度缩放器。
fpSixteen = MixedPrecision(
    param_dtype=torch.float16,
    # Gradient communication precision.
    # 梯度通信的精度。
    reduce_dtype=torch.float16,
    # Buffer precision.
    # 缓冲区的精度。
    buffer_dtype=torch.float16,
)

bfSixteen = MixedPrecision(
    param_dtype=torch.bfloat16,
    # Gradient communication precision.
    # 梯度通信的精度。
    reduce_dtype=torch.bfloat16,
    # Buffer precision.
    # 缓冲区的精度。
    buffer_dtype=torch.bfloat16,
)

bfSixteen_working = MixedPrecision(
    param_dtype=torch.float32,
    reduce_dtype=torch.bfloat16,
    buffer_dtype=torch.bfloat16,
)

fp32_policy = MixedPrecision(
    param_dtype=torch.float32,
    reduce_dtype=torch.float32,
    buffer_dtype=torch.float32,
)
