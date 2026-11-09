# Graph Convolutional Network

图卷积网络（GCN）

This repository contains an implementation of Graph Convolutional Networks (GCN) based on the paper "Semi-Supervised Classification with Graph Convolutional Networks" by Thomas N. Kipf and Max Welling.

本目录实现 Thomas N. Kipf 和 Max Welling 的论文 Semi-Supervised Classification with Graph Convolutional Networks 中提出的图卷积网络。

## Overview

概述

This project implements the GCN model proposed in the paper for semi-supervised node classification on graph-structured data. GCN leverages graph convolutions to aggregate information from neighboring nodes and learn node representations for downstream tasks. The implementation provides a flexible and efficient GCN model for graph-based machine learning tasks.

本项目使用 GCN 对图结构数据进行半监督节点分类。GCN 通过图卷积聚合相邻节点的信息，学习用于后续任务的节点表示。本实现为图机器学习任务提供了灵活、高效的 GCN 模型。

## Requirements

环境依赖

```bash
pip install -r requirements.txt
```

# Usage

使用方法

```bash
python main.py --epochs 200 --lr 0.01 --l2 5e-4 --dropout-p 0.5 --hidden-dim 16 --val-every 20 --include-bias
```

# Dataset

数据集

The implementation includes support for the Cora dataset, a standard benchmark dataset for graph-based machine learning tasks. The Cora dataset consists of scientific publications, where nodes represent papers and edges represent citation relationships. Each paper is associated with a binary label indicating one of seven classes. The dataset is downloaded, preprocessed and ready to use.

本实现支持图机器学习中常用的 Cora 基准数据集。其中，节点代表论文，边代表论文之间的引用关系，每篇论文属于七个类别之一。数据集经过下载和预处理后即可用于模型训练。

## Model Architecture

模型结构

The GCN model architecture follows the details provided in the paper. It consists of multiple graph convolutional layers with ReLU activation, followed by a final softmax layer for classification. The implementation supports customizable hyperparameters such as the number of hidden units, the number of layers, and dropout rate.

模型结构遵循原论文，由带有 ReLU 激活函数的图卷积层和用于分类的最终 softmax 层组成。可调整隐藏单元数量、层数、Dropout 比例等超参数。

# Results

结果

The model achieves a classification accuracy of 82.5% on the test set of the Cora dataset after 200 epochs of training. This result is comparable to the performance reported in the original paper. However, the results can vary due to the randomness of the train/val/test split.

原版报告：训练 200 轮后，模型在 Cora 测试集上达到 82.5% 的分类准确率，与原论文接近。由于训练集、验证集和测试集的划分具有随机性，实际结果可能不同。

References
Thomas N. Kipf and Max Welling. "Semi-Supervised Classification with Graph Convolutional Networks." Link to the paper

参考资料
Thomas N. Kipf 和 Max Welling：Semi-Supervised Classification with Graph Convolutional Networks。

Original paper repository: [https://github.com/tkipf/gcn](https://github.com/tkipf/gcn)

原论文仓库：[tkipf/gcn](https://github.com/tkipf/gcn)。
