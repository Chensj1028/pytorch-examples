"""
This is the script to test 2D Parallel which combines Tensor/Sequence
parallel with Fully Sharded Data Parallel (TP/SP + FSDP) on a example
Llama2 model. We show an E2E working flow from forward, backward
and optimization.

We enabled Fully Sharded Data Parallel + Tensor Parallel in
separate parallel dimensions:
    Data Parallel ("dp") across hosts
    Tensor Parallel ("tp") within each host

 We use a simple diagram to illustrate below:

======================================================================
------------       ------------       ------------       ------------
| Host 1   |       | Host 2   |       |          |       | Host N   |
| 8 GPUs   |       | 8 GPUs   |       |          |       | 8 GPUs   |
|          |       |          |       |    ...   |       |          |
| (TP)     |       | (TP)     |       |          |       | (TP)     |
|[0,1,..,7]|       |[8,9..,15]|       |          |       |[8N-8,8N-7|
|          |       |          |       |          |       | .., 8N-1]|
|          |       |          |       |          |       |          |
------------       ------------       ------------       ------------
FSDP:
[0, 8, ..., 8N-8], [1, 9, ..., 8N-7], ..., [7, 15, ..., 8N-1]
======================================================================

More details can be seen in the PyTorch tutorials:
https://pytorch.org/tutorials/intermediate/TP_tutorial.html
"""
# 此脚本在简化的 Llama2 模型上测试二维并行：将张量/序列并行与完全分片数据并行组合（TP/SP + FSDP）。
# 示例展示前向传播、反向传播和优化更新的完整流程。
# 两个并行维度相互独立：数据并行 dp 跨主机进行，张量并行 tp 在主机内进行。
# 原图中每台主机有 8 个 GPU，主机内的 8 个 GPU 组成 TP 组；
# 各主机上相同本地编号的 GPU 组成 FSDP 组，例如 [0, 8, ..., 8N-8]。
# 更多说明见上方 PyTorch 教程链接。

import sys
import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from log_utils import rank_log, get_logger, verify_min_gpu_count

# ---- GPU check ------------
# 检查 GPU。
_min_gpu_count = 4

if not verify_min_gpu_count(min_gpus=_min_gpu_count):
    print(f"Unable to locate sufficient {_min_gpu_count} gpus to run this example. Exiting.")
    sys.exit()
# ---------------------------

from llama2_model import Transformer, ModelArgs

from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed._tensor import Shard, Replicate
from torch.distributed.tensor.parallel import (
    parallelize_module,
    ColwiseParallel,
    RowwiseParallel,
    PrepareModuleInput,
    SequenceParallel
)

tp_size = 2
logger = get_logger()

# understand world topology
# 确定全局进程拓扑。
_rank = int(os.environ["RANK"])
_world_size = int(os.environ["WORLD_SIZE"])


print(f"Starting PyTorch 2D (FSDP + TP) example on rank {_rank}.")
assert (
    _world_size % tp_size == 0
), f"World size {_world_size} needs to be divisible by TP size {tp_size}"


# create a sharding plan based on the given world_size.
# 根据给定的 world_size 创建分片方案。
dp_size = _world_size // tp_size

device_type = torch.accelerator.current_accelerator().type
# Create a device mesh with 2 dimensions.
# 创建二维设备网格。
# First dim is the data parallel dimension
# 第一维为数据并行维度。
# Second dim is the tensor parallel dimension.
# 第二维为张量并行维度。
device_mesh = init_device_mesh(device_type, (dp_size, tp_size), mesh_dim_names=("dp", "tp"))

rank_log(_rank, logger, f"Device Mesh created: {device_mesh=}")
tp_mesh = device_mesh["tp"]
dp_mesh = device_mesh["dp"]

# For TP, input needs to be same across all TP ranks.
# 对于张量并行（TP），各 TP 进程的输入必须相同。
# while for SP, input can be different across all ranks.
# 对于序列并行（SP），不同进程的输入可以不同。
# We will use dp_rank for setting the random seed
# 使用 dp_rank 设置随机种子，
# to mimic the behavior of the dataloader.
# 模拟数据加载器的行为。
dp_rank = dp_mesh.get_local_rank()

# create model and move it to GPU - initdevice_type_mesh has already mapped GPU ids.
# 创建模型并移动到 GPU；initdevice_type_mesh 已完成 GPU 编号映射。
simple_llama2_config = ModelArgs(dim=256, n_layers=2, n_heads=16, vocab_size=32000)

model = Transformer.from_model_args(simple_llama2_config).to(device_type)

# init model weights
# 初始化模型权重。
model.init_weights()

# parallelize the first embedding and the last linear out projection
# 对最前面的嵌入层和最后的线性输出投影应用并行化。
model = parallelize_module(
    model,
    tp_mesh,
    {
        "tok_embeddings": RowwiseParallel(
            input_layouts=Replicate(),
            output_layouts=Shard(1),
        ),
        "norm": SequenceParallel(),
        "output": ColwiseParallel(
            input_layouts=Shard(1),
            output_layouts=Replicate()
        ),
    }
)

for layer_id, transformer_block in enumerate(model.layers):
    layer_tp_plan = {
        "attention_norm": SequenceParallel(),
        "attention": PrepareModuleInput(
            input_layouts=(Shard(1), Replicate()),
            desired_input_layouts=(Replicate(), Replicate()),
        ),
        "attention.wq": ColwiseParallel(use_local_output=False),
        "attention.wk": ColwiseParallel(use_local_output=False),
        "attention.wv": ColwiseParallel(use_local_output=False),
        "attention.wo": RowwiseParallel(output_layouts=Shard(1)),
        "ffn_norm": SequenceParallel(),
        "feed_forward": PrepareModuleInput(
            input_layouts=(Shard(1),),
            desired_input_layouts=(Replicate(),),
        ),
        "feed_forward.w1": ColwiseParallel(),
        "feed_forward.w2": RowwiseParallel(output_layouts=Shard(1)),
        "feed_forward.w3": ColwiseParallel(),
    }

    # Custom parallelization plan for the model
    # 自定义模型并行化方案。
    parallelize_module(
        module=transformer_block,
        device_mesh=tp_mesh,
        parallelize_plan=layer_tp_plan
    )

# Init FSDP using the dp device mesh
# 使用数据并行设备网格初始化 FSDP。
sharded_model = fully_shard(model, mesh=dp_mesh)

rank_log(_rank, logger, f"Model after parallelization {sharded_model=}\n")

# Create an optimizer for the parallelized and sharded model.
# 为并行化并分片后的模型创建优化器。
lr = 3e-3
rank_log(_rank, logger, f"Creating AdamW optimizer with learning rate {lr}")
optimizer = torch.optim.AdamW(sharded_model.parameters(), lr=lr, foreach=True)

# Training loop:
# 训练循环：
# Perform a num of iterations of forward/backward
# 执行若干次前向传播、反向传播，
# and optimizations for the sharded module.
# 并对分片模块进行优化更新。
rank_log(_rank, logger, "\nStarting 2D training...")
num_iterations = 10
batch_size = 2

for i in range(num_iterations):
    # seeding with dp_rank to ensure identical inputs for TP groups
    # 以 dp_rank 为种子，确保同一 TP 组获得相同输入。
    torch.manual_seed(i + dp_rank)
    inp = torch.randint(32000, (8, 256), device=device_type)

    output = sharded_model(inp)
    output.sum().backward()
    optimizer.step()
    rank_log(_rank, logger, f"2D iter {i} complete")

rank_log(_rank, logger, "2D training successfully completed!")

if dist.is_initialized():
    dist.destroy_process_group()
