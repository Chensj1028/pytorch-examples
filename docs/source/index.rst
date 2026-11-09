PyTorch Examples
================

PyTorch 示例集

This pages lists various PyTorch examples that you can use to learn and
experiment with PyTorch.

本页列出可用于学习和实践 PyTorch 的各类示例。

.. panels::

    Image Classification Using ConvNets
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    **使用卷积网络进行图片分类**

    This example demonstrates how to run image classification
    with `Convolutional Neural Networks ConvNets <https://cs231n.github.io/convolutional-networks/>`__
    on the `MNIST <https://en.wikipedia.org/wiki/MNIST_database>`__ database.

    本例展示如何在 MNIST 数据集上使用卷积神经网络（ConvNet）进行图片分类。

    `GO TO EXAMPLE <https://github.com/pytorch/examples/blob/main/mnist>`__ :opticon:`link-external`

    `查看示例 <https://github.com/pytorch/examples/blob/main/mnist>`__ :opticon:`link-external`

    ---

    Measuring Similarity using Siamese Network
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    **使用孪生网络衡量相似度**

    This example demonstrates how to measure similarity between two images
    using `Siamese network <https://en.wikipedia.org/wiki/Siamese_neural_network>`__
    on the `MNIST <https://en.wikipedia.org/wiki/MNIST_database>`__ database.

    本例展示如何在 MNIST 数据集上使用孪生网络，衡量两张图片的相似度。

    `GO TO EXAMPLE <https://github.com/pytorch/examples/blob/main/siamese_network>`__ :opticon:`link-external`

    `查看示例 <https://github.com/pytorch/examples/blob/main/siamese_network>`__ :opticon:`link-external`

    ---

    Word-level Language Modeling using RNN and Transformer
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    **使用 RNN 和 Transformer 进行词级语言建模**

    This example demonstrates how to train a multi-layer `recurrent neural
    network (RNN) <https://en.wikipedia.org/wiki/Recurrent_neural_network>`__,
    such as Elman, GRU, or LSTM, or Transformer on a language
    modeling task by using the Wikitext-2 dataset.

    本例使用 WikiText-2 数据集训练多层循环神经网络（例如 Elman、GRU、LSTM）或 Transformer，完成语言建模任务。

    `GO TO EXAMPLE <https://github.com/pytorch/examples/blob/main/word_language_model>`__ :opticon:`link-external`

    `查看示例 <https://github.com/pytorch/examples/blob/main/word_language_model>`__ :opticon:`link-external`

    ---

    Training ImageNet Classifiers
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    **训练 ImageNet 分类器**

    This example demonstrates how you can train some of the most popular
    model architectures, including `ResNet <https://en.wikipedia.org/wiki/Residual_neural_network>`__, 
    `AlexNet <https://en.wikipedia.org/wiki/AlexNet>`__, and `VGG <https://arxiv.org/pdf/1409.1556.pdf>`__
    on the `ImageNet <https://image-net.org/>`__ dataset.

    本例展示如何在 ImageNet 数据集上训练 ResNet、AlexNet、VGG 等常见模型。

    `GO TO EXAMPLE <https://github.com/pytorch/examples/blob/main/imagenet>`__ :opticon:`link-external`

    `查看示例 <https://github.com/pytorch/examples/blob/main/imagenet>`__ :opticon:`link-external`

    ---

    Generative Adversarial Networks (DCGAN)
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    **深度卷积生成对抗网络（DCGAN）**

    This example implements the `Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks  <https://arxiv.org/abs/1511.06434>`__ paper.

    本例实现论文《利用深度卷积生成对抗网络进行无监督表示学习》。

    `GO TO EXAMPLE <https://github.com/pytorch/examples/blob/main/dcgan>`__ :opticon:`link-external`

    `查看示例 <https://github.com/pytorch/examples/blob/main/dcgan>`__ :opticon:`link-external`

    ---

    Variational Auto-Encoders
    ^^^^^^^^^^^^^^^^^^^^^^^^^

    **变分自编码器**

    This example implements the `Auto-Encoding Variational Bayes <https://arxiv.org/abs/1312.6114>`__ paper
    with `ReLUs <https://en.wikipedia.org/wiki/Rectifier_(neural_networks)>`__ and the Adam optimizer.

    本例实现论文《自编码变分贝叶斯》，采用 ReLU 和 Adam 优化器。

    `GO TO EXAMPLE <https://github.com/pytorch/examples/blob/main/vae>`__ :opticon:`link-external`

    `查看示例 <https://github.com/pytorch/examples/blob/main/vae>`__ :opticon:`link-external`

    ---

    Super-resolution Using an Efficient Sub-Pixel CNN
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    **使用高效亚像素卷积网络进行超分辨率重建**

    This example demonstrates how to use the sub-pixel convolution layer
    described in `Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network <https://arxiv.org/abs/1609.05158>`__ paper. This example trains a super-resolution
    network on the `BSD300 dataset <https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/>`__.

    本例演示论文《使用高效亚像素卷积神经网络进行实时单图像和视频超分辨率重建》中的亚像素卷积层，并在 BSD300 数据集上训练超分辨率网络。

    `GO TO EXAMPLE <https://github.com/pytorch/examples/blob/main/super_resolution>`__ :opticon:`link-external`

    `查看示例 <https://github.com/pytorch/examples/blob/main/super_resolution>`__ :opticon:`link-external`

    ---

    HOGWILD! Training of Shared ConvNets
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    **共享卷积网络的 HOGWILD! 训练**

    `HOGWILD! <https://arxiv.org/abs/1106.5730>`__ is a scheme that allows
    Stochastic Gradient Descent (SGD)
    parallelization without memory locking. This example demonstrates how
    to perform HOGWILD! training of shared ConvNets on MNIST.

    HOGWILD! 允许在不进行内存加锁的情况下并行执行随机梯度下降（SGD）。本例展示如何在 MNIST 上对共享卷积网络进行 HOGWILD! 训练。

    `GO TO EXAMPLE <https://github.com/pytorch/examples/blob/main/mnist_hogwild>`__ :opticon:`link-external`

    `查看示例 <https://github.com/pytorch/examples/blob/main/mnist_hogwild>`__ :opticon:`link-external`

    ---

    Training a CartPole to balance with actor-critic
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    **使用 Actor-Critic 训练倒立摆保持平衡**

    This reinforcement learning tutorial demonstrates how to train a
    CartPole to balance
    in the `Gymnasium <https://gymnasium.farama.org/>`__ toolkit by using the
    `Actor-Critic <https://proceedings.neurips.cc/paper/1999/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf>`__ method.

    本强化学习教程使用 Actor-Critic 方法，在 Gymnasium 工具包中训练倒立摆保持平衡。

    `GO TO EXAMPLE <https://github.com/pytorch/examples/blob/main/reinforcement_learning>`__ :opticon:`link-external`

    `查看示例 <https://github.com/pytorch/examples/blob/main/reinforcement_learning>`__ :opticon:`link-external`

    ---

    Time Sequence Prediction
    ^^^^^^^^^^^^^^^^^^^^^^^^

    **时间序列预测**

    This beginner example demonstrates how to use LSTMCell to
    learn sine wave signals to predict the signal values in the future.

    本入门示例使用 LSTMCell 学习正弦波信号，并预测未来的信号值。

    `GO TO EXAMPLE <https://github.com/pytorch/examples/tree/main/time_sequence_prediction>`__ :opticon:`link-external`

    `查看示例 <https://github.com/pytorch/examples/tree/main/time_sequence_prediction>`__ :opticon:`link-external`

    ---

    Implement the Neural Style Transfer algorithm on images
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    **对图片应用神经风格迁移算法**

    This tutorial demonstrates how you can use PyTorch's implementation
    of the `Neural Style Transfer (NST) <https://en.wikipedia.org/wiki/Neural_style_transfer>`__
    algorithm on images.

    本教程展示如何使用 PyTorch 实现的神经风格迁移（NST）算法处理图片。

    `GO TO EXAMPLE <https://github.com/pytorch/examples/blob/main/fast_neural_style>`__ :opticon:`link-external`

    `查看示例 <https://github.com/pytorch/examples/blob/main/fast_neural_style>`__ :opticon:`link-external`

    ---

    PyTorch Module Transformations using fx
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    **使用 FX 进行 PyTorch 模块变换**

    This set of examples demonstrates the torch.fx toolkit. For more
    information about `torch.fx`, see
    `torch.fx Overview <https://pytorch.org/docs/master/fx.html>`__.

    这组示例演示 torch.fx 工具包，更多信息可参阅上方链接中的 torch.fx 概述。

    `GO TO EXAMPLE <https://github.com/pytorch/examples/blob/main/fx>`__ :opticon:`link-external`

    `查看示例 <https://github.com/pytorch/examples/blob/main/fx>`__ :opticon:`link-external`

    ---

    Distributed PyTorch
    ^^^^^^^^^^^^^^^^^^^

    **分布式 PyTorch**

    This set of examples demonstrates `Distributed Data Parallel (DDP) <https://pytorch.org/tutorials/intermediate/ddp_tutorial.html>`__ and `Distributed RPC framework <https://pytorch.org/docs/stable/rpc.html>`__. 
    Includes the code used in the `DDP tutorial series <https://pytorch.org/tutorials/beginner/ddp_series_intro.html>`__.

    这组示例演示分布式数据并行（DDP）与分布式 RPC 框架，并包含 DDP 教程系列使用的代码。

    `GO TO EXAMPLES <https://github.com/pytorch/examples/tree/main/distributed>`__ :opticon:`link-external`

    `查看示例 <https://github.com/pytorch/examples/tree/main/distributed>`__ :opticon:`link-external`

    ---

    C++ Frontend
    ^^^^^^^^^^^^

    **C++ 前端**

    The PyTorch C++ frontend is a C++14 library for CPU and GPU tensor computation.
    This set of examples includes a linear regression, autograd, image recognition
    (MNIST), and other useful examples using PyTorch C++ frontend.

    PyTorch C++ 前端是面向 CPU 和 GPU 张量计算的 C++14 库。这组示例包括线性回归、自动求导、MNIST 图片识别及其他实用示例。

    `GO TO EXAMPLES <https://github.com/pytorch/examples/tree/main/cpp>`__ :opticon:`link-external`

    `查看示例 <https://github.com/pytorch/examples/tree/main/cpp>`__ :opticon:`link-external`

    ---

    Image Classification Using Forward-Forward Algorithm
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    **使用 Forward-Forward 算法进行图片分类**

    This example implements the paper `The Forward-Forward Algorithm: Some Preliminary Investigations <https://arxiv.org/pdf/2212.13345.pdf>`__ by Geoffrey Hinton.
    on the `MNIST <https://en.wikipedia.org/wiki/MNIST_database>`__ database.
    It is an introductory example to the Forward-Forward algorithm.

    本例在 MNIST 数据集上实现 Geoffrey Hinton 的论文《Forward-Forward 算法：一些初步探索》，作为该算法的入门示例。

    `GO TO EXAMPLE <https://github.com/pytorch/examples/tree/main/mnist_forward_forward>`__ :opticon:`link-external`

    `查看示例 <https://github.com/pytorch/examples/tree/main/mnist_forward_forward>`__ :opticon:`link-external`

    ---

    Graph Convolutional Network
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^

    **图卷积网络**

    This example implements the `Semi-Supervised Classification with Graph Convolutional Networks <https://arxiv.org/pdf/1609.02907.pdf>`__ paper on the CORA database.

    本例在 Cora 数据集上实现论文《使用图卷积网络进行半监督分类》。

    `GO TO EXAMPLE <https://github.com/pytorch/examples/blob/main/gcn>`__ :opticon:`link-external`

    `查看示例 <https://github.com/pytorch/examples/blob/main/gcn>`__ :opticon:`link-external`
