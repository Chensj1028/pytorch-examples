# PyTorch native Tensor Parallel for distributed training

使用 PyTorch 原生张量并行进行分布式训练

This example demonstrates SPMD Megatron-LM style Tensor Parallel by using
PyTorch native Tensor Parallel APIs, which include:

本示例使用 PyTorch 原生张量并行 API，演示 SPMD Megatron-LM 风格的张量并行，包括：

1. Simple module-level Tensor Parallelism on a dummy MLP model.

   在示例多层感知机（MLP）模型上进行模块级张量并行。

2. Simple module-level Tensor Parallelism with Sequence Parallel inputs/outputs on a dummy MLP model.

   在示例 MLP 模型上进行模块级张量并行，并使用序列并行的输入和输出。

3. A E2E demo of Fully Sharded Data Parallel + Tensor Parallel (with Sequence Parallel) on a example Llama2 model.

   在示例 Llama2 模型上，将完全分片数据并行（FSDP）与张量并行、序列并行结合的端到端演示。

More details about the PyTorch native Tensor Parallel APIs, please see PyTorch docs:
https://pytorch.org/docs/stable/distributed.tensor.parallel.html

API 详情见 [PyTorch 原生张量并行文档](https://pytorch.org/docs/stable/distributed.tensor.parallel.html)。

## Installation

安装

```bash
pip install -r requirements.txt
```

## Running Examples

运行示例

You can run the examples using `torchrun` to launch distributed training:

使用 `torchrun` 启动分布式训练：

```bash
# Simple Tensor Parallel example
# 简单张量并行示例。
torchrun --nnodes=1 --nproc_per_node=4 tensor_parallel_example.py

# Tensor Parallel with Sequence Parallel
# 结合序列并行的张量并行。
torchrun --nnodes=1 --nproc_per_node=4 sequence_parallel_example.py

# FSDP + Tensor Parallel with Llama2 model
# 在 Llama2 模型上结合 FSDP 与张量并行。
torchrun --nnodes=1 --nproc_per_node=4 fsdp_tp_example.py
```

For more details, check the `run_examples.sh` script.

更多细节见 `run_examples.sh`。
