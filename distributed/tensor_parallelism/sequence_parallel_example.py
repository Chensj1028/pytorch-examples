"""
This is the script to test Sequence Parallel(SP) on a toy model in a
Megetron-LM SPMD style. We show an E2E working flow from forward,
backward and optimization.

We use the example of two `nn.Linear` layers with an element-wise `nn.RELU`
in between to show an example of sequence parallel, which was proposed in paper:

https://arxiv.org/pdf/2205.05198.pdf.

Like tensor parallel, we parallelize the first linear layer by column
and also parallelize the second linear layer by row. But the input in each rank
now is different so that we need one all-gather for input and one reduce-scatter
in the end of the second linear layer.

The following is an example command to run this code
    torchrun --nnodes 1 --nproc-per-node 4 sequence_parallel_example.py
"""
# 此脚本以 Megatron-LM 的 SPMD 风格，在简化模型上演示序列并行（SP），涵盖前向、反向和优化更新。
# 模型包含两个 nn.Linear，中间为逐元素 ReLU；序列并行方法见原文所列论文。
# 第一层按列并行，第二层按行并行。由于各进程的输入不同，需要先 all-gather 输入，
# 再在第二个线性层末尾执行 reduce-scatter。
# 运行命令：torchrun --nnodes 1 --nproc-per-node 4 sequence_parallel_example.py。

import os
import sys
import torch
import torch.nn as nn

import torch.distributed as dist
from torch.distributed._tensor import Shard

from torch.distributed.tensor.parallel import (
    parallelize_module,
    ColwiseParallel,
    RowwiseParallel,
)

from log_utils import rank_log, get_logger, verify_min_gpu_count


# ---- GPU check ------------
# 检查 GPU。
_min_gpu_count = 2

if not verify_min_gpu_count(min_gpus=_min_gpu_count):
    print(f"Unable to locate sufficient {_min_gpu_count} gpus to run this example. Exiting.")
    sys.exit()
# ---------------------------

from torch.distributed._tensor.device_mesh import init_device_mesh

class ToyModel(nn.Module):
    """MLP based model"""
    # 基于多层感知机（MLP）的模型。

    def __init__(self):
        super().__init__()
        self.in_proj = nn.Linear(10, 32)
        self.relu = nn.ReLU()
        self.out_proj = nn.Linear(32, 5)

    def forward(self, x):
        return self.out_proj(self.relu(self.in_proj(x)))


"""
Main body of the demo of a basic version of sequence parallel by using
PyTorch native APIs.
"""
# 使用 PyTorch 原生 API 演示基础序列并行的主体代码。
logger = get_logger()

device_type = torch.accelerator.current_accelerator().type
# create a device mesh based on the given world_size.
# 根据给定的 world_size 创建设备网格。
device_mesh = init_device_mesh(
    device_type=device_type, mesh_shape=(int(os.environ["WORLD_SIZE"]),)
)

_rank = device_mesh.get_rank()

print(f"Starting PyTorch Sequence Parallel example on rank {_rank}.")

rank_log(_rank, logger, f"Device Mesh created: {device_mesh=}")

# create model and move it to GPU.  Init_device_mesh has already assigned gpu ids...
# 创建模型并移动到 GPU；init_device_mesh 已分配 GPU 编号。
model = ToyModel().to(device_type)

# Custom parallelization plan for the model
# 自定义模型并行化方案。
sp_model = parallelize_module(
    module=model,
    device_mesh=device_mesh,
    parallelize_plan={
        "in_proj": ColwiseParallel(input_layouts=Shard(0)),
        "out_proj": RowwiseParallel(output_layouts=Shard(0)),
    },
)


# Create a optimizer for the parallelized module.
# 为并行化后的模块创建优化器。
lr = 0.25
optimizer = torch.optim.AdamW(sp_model.parameters(), lr=lr, foreach=True)


# Perform a num of iterations of forward/backward
# 执行若干次前向传播、反向传播，
# and optimizations for the sharded module.
# 并对分片模块进行优化更新。
num_iters = 10
rank_log(_rank, logger, "Sequence Parallel training starting...")

for i in range(num_iters):
    # For SP, input can be different across all ranks.
    # 对于序列并行，各进程的输入可以不同。
    inp = torch.rand(20, 10, device=device_type)
    output = sp_model(inp)
    output.sum().backward()
    optimizer.step()
    rank_log(_rank, logger, f"Sequence Parallel iter {i} completed")

rank_log(_rank, logger, "Sequence Parallel training completed!")

if dist.is_initialized():
    dist.destroy_process_group()
