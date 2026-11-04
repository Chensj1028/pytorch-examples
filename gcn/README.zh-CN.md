# 图卷积网络（GCN）

[English original](README.md)

> 中文说明；代码块中的命令、代码及程序输出保留原样。

本目录实现 Thomas N. Kipf 和 Max Welling 的论文 Semi-Supervised Classification with Graph Convolutional Networks 中提出的图卷积网络。

## 概述

本项目使用 GCN 对图结构数据进行半监督节点分类。GCN 通过图卷积聚合相邻节点的信息，学习用于后续任务的节点表示。本实现为图机器学习任务提供了灵活、高效的 GCN 模型。

## 环境依赖

```bash
pip install -r requirements.txt
```

## 使用方法

```bash
python main.py --epochs 200 --lr 0.01 --l2 5e-4 --dropout-p 0.5 --hidden-dim 16 --val-every 20 --include-bias
```

## 数据集

本实现支持图机器学习中常用的 Cora 基准数据集。其中，节点代表论文，边代表论文之间的引用关系，每篇论文属于七个类别之一。数据集经过下载和预处理后即可用于模型训练。

## 模型结构

模型结构遵循原论文，由带有 ReLU 激活函数的图卷积层和用于分类的最终 softmax 层组成。可调整隐藏单元数量、层数、Dropout 比例等超参数。

## 结果

原版报告：训练 200 轮后，模型在 Cora 测试集上达到 82.5% 的分类准确率，与原论文接近。由于训练集、验证集和测试集的划分具有随机性，实际结果可能不同。

## 参考资料

Thomas N. Kipf 和 Max Welling：Semi-Supervised Classification with Graph Convolutional Networks。

原论文仓库：[tkipf/gcn](https://github.com/tkipf/gcn)。
