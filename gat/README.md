# Graph Attention Network

图注意力网络（GAT）

This repository contains a PyTorch implementation of the **Graph Attention Networks (GAT)** based on the paper ["Graph Attention Network" by Velickovic et al](https://arxiv.org/abs/1710.10903v3). 

本目录基于 [Veličković 等人的 Graph Attention Networks 论文](https://arxiv.org/abs/1710.10903v3)，提供图注意力网络（GAT）的 PyTorch 实现。

The Graph Attention Network is a powerful graph neural network model for learning represtations on graph-structured data, which has shown excellent performance in various tasks such as node classification, link prediction, and graph classification.

GAT 用于学习图结构数据的表示，在节点分类、链接预测、图分类等任务中表现良好。

## Overview

概述

The Graph Attention Network (GAT) is a graph neural network architecture designed specifically for handling graph-structured data. It leverages multi-head attention mechanism to capture the information of neighboring nodes in an attentive manner to learn represtations for each node. This attention mechanism allows the model to focus on relevant nodes and adaptively weight their contributions during message passing.

图注意力网络专门处理图结构数据，利用多头注意力机制聚合相邻节点的信息，为每个节点学习表示。注意力机制使模型能够关注相关节点，并在消息传递时自适应调整各邻居的贡献权重。

Check out the following resources for more ino on GATs:

更多资料：

- [Blog post by the main auther, Petar Velickovic](https://petar-v.com/GAT/)

  [第一作者 Petar Veličković 的博客](https://petar-v.com/GAT/)

- [Main paper](https://doi.org/10.48550/arXiv.1710.10903)

  [原论文](https://doi.org/10.48550/arXiv.1710.10903)

This repository provides a clean and short implementation of the official GAT model using PyTorch. The code is well-documented and easy to understand, making it a valuable resource for researchers and practitioners interested in graph deep learning.

本目录用 PyTorch 简洁实现官方 GAT 模型，并提供说明，供希望学习图深度学习的研究者和实践者参考。

## Key Features

主要功能

- **GAT Model**: Implementation of the Graph Attention Network model with multi-head attention based on the paper "Graph Attention Network" by Velickovic et al.

  **GAT 模型**：实现论文中的多头注意力图神经网络。

- **Graph Attention Layers**: Implementation of graph convolutional layers that aggregate information from neighboring nodes using a self-attention mechanisms to learn node importance weights.

  **图注意力层**：通过自注意力机制学习邻居节点的重要性权重，并聚合邻居信息。

- **Training and Evaluation**: Code for training GAT models on graph-structured data and evaluating their performance on node classification tasks on the *Cora* benchmark dataset.

  **训练与评估**：提供在 Cora 基准数据集上训练模型、评估节点分类效果的代码。

---

# Requirements

环境依赖

- Python 3.7 or higher

  Python 3.7 或更高版本

- PyTorch 2.0 or higher

  PyTorch 2.0 或更高版本

- Requests 2.31 or higher

  Requests 2.31 或更高版本

- NumPy 1.24 or higher

  NumPy 1.24 或更高版本

# Dataset

数据集

The implementation includes support for the Cora dataset, a standard benchmark dataset for graph-based machine learning tasks. The Cora dataset consists of scientific publications, where nodes represent papers and edges represent citation relationships. Each paper is associated with a binary label indicating one of seven classes. The dataset is downloaded, preprocessed and ready to use.

本实现支持 Cora 数据集，这是图机器学习中常用的基准数据集。节点代表论文，边代表论文之间的引用关系，每篇论文属于七个类别之一。数据集经过下载和预处理后即可使用。

# Model Architecture

模型结构

The official architecture (used in this project) proposed in the paper "Graph Attention Network" by Velickovic et al. consists of two graph attention layers which incorporates the multi-head attention mechanisms during its message trasformation and aggregation. Each graph attention layer applies a shared self-attention mechanism to every node in the graph, allowing them to learn different representations based on the importance of their neighbors.

本项目采用论文中的官方结构，由两个图注意力层组成，在消息变换和聚合中使用多头注意力机制。图注意力层对各节点应用共享的自注意力机制，根据邻居的重要性学习节点表示。

In terms of activation functions, the GAT model employs both the **Exponential Linear Unit (ELU)** and the **Leaky Rectified Linear Unit (LeakyReLU)** activations, which introduce non-linearity to the model. ELU is used as the activation function for the **hidden layers**, while LeakyReLU is applied to the **attention coefficients** to ensure non-zero gradients for negative values.

激活函数包括 **ELU（指数线性单元）** 和 **LeakyReLU（带泄漏的整流线性单元）**。隐藏层使用 ELU；注意力系数计算使用 LeakyReLU，使负值区域也具有非零梯度。

Following the official implementation, the first GAT layer consists of **K = 8 attention heads** computing **F' = 8 features** each (for a **total of 64 features**) followed by an exponential linear unit (ELU) activation on the layer outputs. The second GAT layer is used for classification: a **single attention head** that computes C features (where C is the number of classes), followed by a softmax activation for probablisitic outputs. (we use log-softmax instead for computational convenience with using NLLLoss)

按照官方实现，第一层有 **K = 8 个注意力头**，每个头输出 **F' = 8 个特征**，合计 **64 个特征**，随后应用 ELU。第二层用于分类，只有**一个注意力头**，输出 C 个特征，C 为类别数量，再使用 softmax 得到类别概率。本实现为便于配合 `NLLLoss`，改用 log-softmax。

*Note that due to being an educational example, this implementation uses the full dense form of the adjacency matrix of the graph, and not the sparse form of the matrix. Thus all the operations in the model implemeation is done in a non-sparse from. This will not affect the model's performance accuracy-wise. However an sparse-friendly implementation will help with the efficiency in the use of resources, storage, and speed.*

> 作为教学示例，本实现使用完整的稠密邻接矩阵，计算也采用稠密形式。按照原版说明，这不会影响模型的分类准确率，但使用适合稀疏数据的实现，可以改善资源使用、存储和运行速度。

# Usage

使用方法

Training and evaluating the GAT model on the Cora dataset can be done through running the `main.py` script as follows:

运行 `main.py`，在 Cora 数据集上训练和评估模型。

1. Clone the PyTorch examples repository:

   克隆 PyTorch 示例仓库：

```
git clone https://github.com/pytorch/examples.git
cd examples/gat
```

2. Install the required dependencies:

   安装依赖：

```
pip install -r requirements.txt
```

3. Train the GAT model by running the `main.py` script as follows:: (Example using the default parameters)

   使用默认参数训练 GAT 模型：

```bash
python main.py --epochs 300 --lr 0.005 --l2 5e-4 --dropout-p 0.6 --num-heads 8 --hidden-dim 64 --val-every 20
```

In more detail, the `main.py` script recieves following arguments:

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

# Results

结果

After training for **300 epochs** with default hyperparameters on random train/val/test data splits, the GAT model achieves around **%81.25** classification accuracy on the test split. This result is comparable to the performance reported in the original paper. However, the results can vary due to the randomness of the train/val/test split.

原版报告：使用默认超参数、随机划分训练集/验证集/测试集，训练 **300 轮**后，测试集分类准确率约为 **81.25%**，与原论文接近。实际结果会随数据划分的随机性而变化。

# Reference

参考资料

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

- Paper on arxiv: [arXiv:1710.10903v3](https://doi.org/10.48550/arXiv.1710.10903)

  arXiv 论文：[arXiv:1710.10903v3](https://doi.org/10.48550/arXiv.1710.10903)

- Original paper repository: [https://github.com/PetarV-/GAT](https://github.com/PetarV-/GAT)

  原论文仓库：[PetarV-/GAT](https://github.com/PetarV-/GAT)
