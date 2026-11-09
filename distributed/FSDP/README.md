Note: FSDP1 is deprecated. Please follow [FSDP2 tutorial](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html) and [code examples](https://github.com/pytorch/examples/tree/main/distributed/FSDP2).

> FSDP1 已弃用，请参考 [FSDP2 教程](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html)和[代码示例](https://github.com/pytorch/examples/tree/main/distributed/FSDP2)。本地说明见 [FSDP2 说明](../FSDP2/README.md)。

## FSDP1 T5

FSDP1 T5 示例

To run the T5 example with FSDP1 for text summarization:

下面介绍如何使用 FSDP1 运行 T5 文本摘要示例。

## Get the wikihow dataset

获取 WikiHow 数据集

```bash

sh download_dataset.sh

```

## Install the requirements:

安装依赖

~~~
pip install -r requirements.txt
~~~

## Ensure you are running a recent version of PyTorch:

PyTorch 版本

see https://pytorch.org to install at least 1.12 and ideally a current nightly build. 

原版要求至少使用 PyTorch 1.12，并建议使用当时较新的 nightly 构建。安装方式见 https://pytorch.org 。

Start the training with Torchrun (adjust nproc_per_node to your GPU count):

使用 Torchrun 启动训练，将 `nproc_per_node` 调整为实际 GPU 数量：

```
torchrun --nnodes 1 --nproc_per_node 4  T5_training.py

```
