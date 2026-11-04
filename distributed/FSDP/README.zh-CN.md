# FSDP1 T5 示例

[English original](README.md)

> 中文说明；代码块中的命令、代码及程序输出保留原样。

> FSDP1 已弃用，请参考 [FSDP2 教程](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html)和[代码示例](https://github.com/pytorch/examples/tree/main/distributed/FSDP2)。本地中文版见 [FSDP2 说明](../FSDP2/README.zh-CN.md)。

下面介绍如何使用 FSDP1 运行 T5 文本摘要示例。

## 获取 WikiHow 数据集

```bash

sh download_dataset.sh

```

## 安装依赖

~~~
pip install -r requirements.txt
~~~

## PyTorch 版本

原版要求至少使用 PyTorch 1.12，并建议使用当时较新的 nightly 构建。安装方式见 https://pytorch.org 。

使用 Torchrun 启动训练，将 `nproc_per_node` 调整为实际 GPU 数量：

```
torchrun --nnodes 1 --nproc_per_node 4  T5_training.py

```
