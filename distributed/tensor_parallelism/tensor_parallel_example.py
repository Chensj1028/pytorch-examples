"""
This is the script to test Tensor Parallel(TP) on a toy model in a
Megetron-LM SPMD style. We show an E2E working flow from forward,
backward and optimization.

More context about API designs can be found in the design:

https://github.com/pytorch/pytorch/issues/89884.

And it is built on top of Distributed Tensor which is proposed in:

https://github.com/pytorch/pytorch/issues/88838.

We use the example of two `nn.Linear` layers with an element-wise `nn.RELU`
in between to show an example of Megatron-LM, which was proposed in paper:

https://arxiv.org/abs/1909.08053.

The basic idea is that we parallelize the first linear layer by column
and also parallelize the second linear layer by row so that we only need
one all reduce in the end of the second linear layer.

We can speed up the model training by avoiding communications between
two layers.

To parallelize a nn module, we need to specify what parallel style we want
to use and our `parallelize_module` API will parse and parallelize the modules
based on the given `ParallelStyle`. We are using this PyTorch native Tensor
Parallelism APIs in this example to show users how to use them.

The following is an example command to run this code
    torchrun --nnodes 1 --nproc-per-node 4 tensor_parallel_example.py
"""
# 此脚本以 Megatron-LM 的 SPMD 风格，在简化模型上演示张量并行（TP），涵盖前向、反向和优化更新。
# API 设计及底层分布式张量方案见原文所列的两个问题讨论；Megatron-LM 方法见原文论文。
# 示例含两个 nn.Linear，中间为逐元素 ReLU。第一层按列并行，第二层按行并行，
# 因此只需在第二层末尾执行一次 all-reduce，避免两层之间的通信以加快训练。
# 指定 ParallelStyle 后，parallelize_module 会解析并并行化模块。
# 本例使用 PyTorch 原生张量并行 API 展示具体用法。
# 运行命令：torchrun --nnodes 1 --nproc-per-node 4 tensor_parallel_example.py。

import os
import sys
import torch
import torch.nn as nn
import torch.distributed as dist
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
        super(ToyModel, self).__init__()
        self.in_proj = nn.Linear(10, 32)
        self.relu = nn.ReLU()
        self.out_proj = nn.Linear(32, 5)

    def forward(self, x):
        return self.out_proj(self.relu(self.in_proj(x)))


"""
Main body of the demo of a basic version of tensor parallel by using
PyTorch native APIs.
"""
# 使用 PyTorch 原生 API 演示基础张量并行的主体代码。
logger = get_logger()

# create a device mesh based on the given world_size.
# 根据给定的 world_size 创建设备网格。
_world_size = int(os.environ["WORLD_SIZE"])
device_type = torch.accelerator.current_accelerator().type
device_mesh = init_device_mesh(device_type=device_type, mesh_shape=(_world_size,))
_rank = device_mesh.get_rank()


print(f"Starting PyTorch TP example on rank {_rank}.")
assert (
    _world_size % 2 == 0
), f"TP examples require even number of GPUs, but got {_world_size} gpus"

rank_log(_rank, logger, f"Device Mesh created: {device_mesh=}")

# create model and move it to GPU - initdevice_type_mesh has already mapped GPU ids.
# 创建模型并移动到 GPU；initdevice_type_mesh 已完成 GPU 编号映射。
tp_model = ToyModel().to(device_type)


# Custom parallelization plan for the model
# 自定义模型并行化方案。
tp_model = parallelize_module(
    module=tp_model,
    device_mesh=device_mesh,
    parallelize_plan={
        "in_proj": ColwiseParallel(),
        "out_proj": RowwiseParallel(),
    },
)

# Create an optimizer for the parallelized module.
# 为并行化模块创建优化器。
lr = 0.25
optimizer = torch.optim.AdamW(tp_model.parameters(), lr=lr, foreach=True)


# Perform a num of iterations of forward/backward
# 执行若干次前向传播、反向传播，
# and optimizations for the sharded module.
# 并对分片模块进行优化更新。
num_iters = 10
rank_log(_rank, logger, "Tensor Parallel training starting...")

for i in range(num_iters):
    # For TP, input needs to be same across all TP ranks.
    # 对于张量并行（TP），各 TP 进程的输入必须相同。
    # Setting the random seed is to mimic the behavior of dataloader.
    # 设置随机种子，以模拟数据加载器的行为。
    torch.manual_seed(i)
    inp = torch.rand(20, 10, device=device_type)
    output = tp_model(inp)
    output.sum().backward()
    optimizer.step()
    rank_log(_rank, logger, f"Tensor Parallel iter {i} completed")

rank_log(_rank, logger, "Tensor Parallel training completed!")

if dist.is_initialized():
    dist.destroy_process_group()
