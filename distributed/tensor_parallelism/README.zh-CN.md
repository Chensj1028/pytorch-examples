# 使用 PyTorch 原生张量并行进行分布式训练

[English original](README.md)

> 中文说明；代码块中的命令、代码及程序输出保留原样。

本示例使用 PyTorch 原生张量并行 API，演示 SPMD Megatron-LM 风格的张量并行，包括：

1. 在示例多层感知机（MLP）模型上进行模块级张量并行。
2. 在示例 MLP 模型上进行模块级张量并行，并使用序列并行的输入和输出。
3. 在示例 Llama2 模型上，将完全分片数据并行（FSDP）与张量并行、序列并行结合的端到端演示。

API 详情见 [PyTorch 原生张量并行文档](https://pytorch.org/docs/stable/distributed.tensor.parallel.html)。

## 安装

```bash
pip install -r requirements.txt
```

## 运行示例

使用 `torchrun` 启动分布式训练：

```bash
# Simple Tensor Parallel example
torchrun --nnodes=1 --nproc_per_node=4 tensor_parallel_example.py

# Tensor Parallel with Sequence Parallel
torchrun --nnodes=1 --nproc_per_node=4 sequence_parallel_example.py

# FSDP + Tensor Parallel with Llama2 model
torchrun --nnodes=1 --nproc_per_node=4 fsdp_tp_example.py
```

更多细节见 `run_examples.sh`。
