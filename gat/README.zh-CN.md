# 图注意力网络（GAT）

[English original](README.md)

> 中文说明；代码块中的命令、代码及程序输出保留原样。

本目录基于 [Veličković 等人的 Graph Attention Networks 论文](https://arxiv.org/abs/1710.10903v3)，提供图注意力网络（GAT）的 PyTorch 实现。

GAT 用于学习图结构数据的表示，在节点分类、链接预测、图分类等任务中表现良好。

## 概述

图注意力网络专门处理图结构数据，利用多头注意力机制聚合相邻节点的信息，为每个节点学习表示。注意力机制使模型能够关注相关节点，并在消息传递时自适应调整各邻居的贡献权重。

更多资料：

- [第一作者 Petar Veličković 的博客](https://petar-v.com/GAT/)
- [原论文](https://doi.org/10.48550/arXiv.1710.10903)

本目录用 PyTorch 简洁实现官方 GAT 模型，并提供说明，供希望学习图深度学习的研究者和实践者参考。

## 主要功能

- **GAT 模型**：实现论文中的多头注意力图神经网络。
- **图注意力层**：通过自注意力机制学习邻居节点的重要性权重，并聚合邻居信息。
- **训练与评估**：提供在 Cora 基准数据集上训练模型、评估节点分类效果的代码。

---

## 环境依赖

- Python 3.7 或更高版本
- PyTorch 2.0 或更高版本
- Requests 2.31 或更高版本
- NumPy 1.24 或更高版本

## 数据集

本实现支持 Cora 数据集，这是图机器学习中常用的基准数据集。节点代表论文，边代表论文之间的引用关系，每篇论文属于七个类别之一。数据集经过下载和预处理后即可使用。

## 模型结构

本项目采用论文中的官方结构，由两个图注意力层组成，在消息变换和聚合中使用多头注意力机制。图注意力层对各节点应用共享的自注意力机制，根据邻居的重要性学习节点表示。

激活函数包括 **ELU（指数线性单元）** 和 **LeakyReLU（带泄漏的整流线性单元）**。隐藏层使用 ELU；注意力系数计算使用 LeakyReLU，使负值区域也具有非零梯度。

按照官方实现，第一层有 **K = 8 个注意力头**，每个头输出 **F' = 8 个特征**，合计 **64 个特征**，随后应用 ELU。第二层用于分类，只有**一个注意力头**，输出 C 个特征，C 为类别数量，再使用 softmax 得到类别概率。本实现为便于配合 `NLLLoss`，改用 log-softmax。

> 作为教学示例，本实现使用完整的稠密邻接矩阵，计算也采用稠密形式。按照原版说明，这不会影响模型的分类准确率，但使用适合稀疏数据的实现，可以改善资源使用、存储和运行速度。

## 使用方法

运行 `main.py`，在 Cora 数据集上训练和评估模型。

1. 克隆 PyTorch 示例仓库：

```
git clone https://github.com/pytorch/examples.git
cd examples/gat
```

2. 安装依赖：

```
pip install -r requirements.txt
```

3. 使用默认参数训练 GAT 模型：

```bash
python main.py --epochs 300 --lr 0.005 --l2 5e-4 --dropout-p 0.6 --num-heads 8 --hidden-dim 64 --val-every 20
```

`main.py` 支持的参数如下：

```
usage: main.py [-h] [--epochs EPOCHS] [--lr LR] [--l2 L2] [--dropout-p DROPOUT_P] [--hidden-dim HIDDEN_DIM] [--num-heads NUM_HEADS] [--concat-heads] [--val-every VAL_EVERY]
               [--no-cuda] [--no-mps] [--dry-run] [--seed S]

PyTorch Graph Attention Network

options:
  -h, --help            show this help message and exit
  --epochs EPOCHS       number of epochs to train (default: 300)
  --lr LR               learning rate (default: 0.005)
  --l2 L2               weight decay (default: 6e-4)
  --dropout-p DROPOUT_P
                        dropout probability (default: 0.6)
  --hidden-dim HIDDEN_DIM
                        dimension of the hidden representation (default: 64)
  --num-heads NUM_HEADS
                        number of the attention heads (default: 4)
  --concat-heads        wether to concatinate attention heads, or average over them (default: False)
  --val-every VAL_EVERY
                        epochs to wait for print training and validation evaluation (default: 20)
  --no-accel            disables accelerator
  --dry-run             quickly check a single pass
  --seed S              random seed (default: 13)
```

## 结果

原版报告：使用默认超参数、随机划分训练集/验证集/测试集，训练 **300 轮**后，测试集分类准确率约为 **81.25%**，与原论文接近。实际结果会随数据划分的随机性而变化。

## 参考资料

```
@article{
  velickovic2018graph,
  title="{Graph Attention Networks}",
  author={Veli{\v{c}}kovi{\'{c}}, Petar and Cucurull, Guillem and Casanova, Arantxa and Romero, Adriana and Li{\`{o}}, Pietro and Bengio, Yoshua},
  journal={International Conference on Learning Representations},
  year={2018},
  url={https://openreview.net/forum?id=rJXMpikCZ},
}
```

- arXiv 论文：[arXiv:1710.10903v3](https://doi.org/10.48550/arXiv.1710.10903)
- 原论文仓库：[PetarV-/GAT](https://github.com/PetarV-/GAT)
