# PyTorch Examples

PyTorch 示例集

English explanations appear first, followed immediately by Chinese translations. Source comments use the same order.

英文说明在上，中文翻译紧接其下；源码注释也采用相同顺序。

https://pytorch.org/examples/

项目主页：https://pytorch.org/examples/

`pytorch/examples` is a repository showcasing examples of using [PyTorch](https://github.com/pytorch/pytorch). The goal is to have curated, short, few/no dependencies _high quality_ examples that are substantially different from each other that can be emulated in your existing work.

`pytorch/examples` 展示 [PyTorch](https://github.com/pytorch/pytorch) 的使用示例。项目旨在提供经过整理、简短、依赖较少或无需额外依赖的高质量示例，覆盖不同任务，方便在自己的项目中参考使用。

- For tutorials: https://github.com/pytorch/tutorials

  教程：[pytorch/tutorials](https://github.com/pytorch/tutorials)。

- For changes to pytorch.org: https://github.com/pytorch/pytorch.github.io

  PyTorch 官网内容修改：[pytorch/pytorch.github.io](https://github.com/pytorch/pytorch.github.io)。

- For a general model hub: https://pytorch.org/hub/ or https://huggingface.co/models

  通用模型中心：[PyTorch Hub](https://pytorch.org/hub/) 或 [Hugging Face 模型库](https://huggingface.co/models)。

- For recipes on how to run PyTorch in production: https://github.com/facebookresearch/recipes

  生产环境中的 PyTorch 使用方案：[facebookresearch/recipes](https://github.com/facebookresearch/recipes)。

- For general Q&A and support: https://discuss.pytorch.org/

  问答与支持：[PyTorch 论坛](https://discuss.pytorch.org/)。

## Available models

示例与模型

- [Image classification (MNIST) using Convnets](./mnist/README.md)

  [使用卷积网络进行 MNIST 图片分类](./mnist/README.md)

- [Word-level Language Modeling using RNN and Transformer](./word_language_model/README.md)

  [使用 RNN 和 Transformer 进行词级语言建模](./word_language_model/README.md)

- [Training Imagenet Classifiers with Popular Networks](./imagenet/README.md)

  [使用常见网络训练 ImageNet 分类器](./imagenet/README.md)

- [Generative Adversarial Networks (DCGAN)](./dcgan/README.md)

  [深度卷积生成对抗网络（DCGAN）](./dcgan/README.md)

- [Variational Auto-Encoders](./vae/README.md)

  [变分自编码器（VAE）](./vae/README.md)

- [Superresolution using an efficient sub-pixel convolutional neural network](./super_resolution/README.md)

  [使用高效亚像素卷积网络实现超分辨率](./super_resolution/README.md)

- [Hogwild training of shared ConvNets across multiple processes on MNIST](mnist_hogwild)

  [使用 Hogwild 在多个进程间共享卷积网络，训练 MNIST 模型](mnist_hogwild/README.md)

- [Training a CartPole to balance with actor-critic](./reinforcement_learning/README.md)

  [使用 Actor-Critic 训练 CartPole 保持平衡](./reinforcement_learning/README.md)

- [Natural Language Inference (SNLI) with GloVe vectors, LSTMs, and torchtext](legacy/snli)

  [结合 GloVe 词向量、LSTM 和 torchtext 进行 SNLI 自然语言推理](legacy/snli/README.md)

- [Time sequence prediction - use an LSTM to learn Sine waves](./time_sequence_prediction/README.md)

  [使用 LSTM 学习正弦波，预测时间序列](./time_sequence_prediction/README.md)

- [Implement the Neural Style Transfer algorithm on images](./fast_neural_style/README.md)

  [对图片进行神经风格迁移](./fast_neural_style/README.md)

- [Reinforcement Learning with Actor Critic and REINFORCE algorithms on OpenAI gym](./reinforcement_learning/README.md)

  [在 OpenAI Gym 中使用 Actor-Critic 和 REINFORCE 进行强化学习](./reinforcement_learning/README.md)

- [PyTorch Module Transformations using fx](./fx/README.md)

  [使用 FX 进行 PyTorch 模块变换](./fx/README.md)

- Distributed PyTorch examples with [Distributed Data Parallel](./distributed/ddp/README.md) and [RPC](./distributed/rpc)

  分布式 PyTorch 示例：[分布式数据并行（DDP）](./distributed/ddp/README.md)和 [RPC](./distributed/rpc)。

- [Several examples illustrating the C++ Frontend](cpp)

  [PyTorch C++ 前端示例](cpp)

- [Image Classification Using Forward-Forward](./mnist_forward_forward/README.md)

  [使用 Forward-Forward 进行图片分类](./mnist_forward_forward/README.md)

- [Language Translation using Transformers](./language_translation/README.md)

  [使用 Transformer 进行语言翻译](./language_translation/README.md)

Additionally, a list of good examples hosted in their own repositories:

其他独立仓库中的示例：

- [Neural Machine Translation using sequence-to-sequence RNN with attention (OpenNMT)](https://github.com/OpenNMT/OpenNMT-py)

  [使用带注意力机制的序列到序列 RNN 进行神经机器翻译（OpenNMT）](https://github.com/OpenNMT/OpenNMT-py)

## Contributing

参与贡献

If you'd like to contribute your own example or fix a bug please make sure to take a look at [CONTRIBUTING.md](CONTRIBUTING.md).

如果想提交自己的示例或修复问题，请先阅读 [CONTRIBUTING.md](CONTRIBUTING.md)。

## Getting started

初学者从这里开始


建议先阅读 [MNIST 手写数字识别](./mnist/README.md)，再结合本地的[入门说明](./docs/入门说明.md)和 `mnist/main.py` 学习数据加载、模型、训练及测试。

复习整体过程可阅读 [MNIST 学习笔记与流程图](./docs/mnist-learning-notes.zh-CN.md)。


## Complete example index

完整示例索引

全部 40 份 README 已采用同页中英对照；下表列出其余 39 份：

| 示例目录 | 中文说明 |
| --- | --- |
| `cpp/autograd` | [C++ 自动求导示例](cpp/autograd/README.md) |
| `cpp/custom-dataset` | [使用 PyTorch C++ 前端加载自定义数据集](cpp/custom-dataset/README.md) |
| `cpp/dcgan` | [使用 PyTorch C++ 前端训练 DCGAN](cpp/dcgan/README.md) |
| `cpp/distributed` | [使用 PyTorch C++ 前端（LibTorch）在 MNIST 上进行分布式训练](cpp/distributed/README.md) |
| `cpp/mnist` | [使用 PyTorch C++ 前端识别 MNIST 手写数字](cpp/mnist/README.md) |
| `cpp/regression` | [C++ 线性回归示例](cpp/regression/README.md) |
| `cpp/transfer-learning` | [使用 LibTorch 和 OpenCV 在猫狗数据集上进行迁移学习](cpp/transfer-learning/README.md) |
| `dcgan` | [深度卷积生成对抗网络（DCGAN）](dcgan/README.md) |
| `distributed/ddp` | [使用 PyTorch 构建分布式数据并行（DDP）应用](distributed/ddp/README.md) |
| `distributed/ddp-tutorial-series` | [PyTorch 分布式训练教程系列](distributed/ddp-tutorial-series/README.md) |
| `distributed/FSDP` | [FSDP1 T5 示例](distributed/FSDP/README.md) |
| `distributed/FSDP2` | [FSDP2 示例](distributed/FSDP2/README.md) |
| `distributed/minGPT-ddp` | [minGPT 分布式数据并行训练](distributed/minGPT-ddp/README.md) |
| `distributed/rpc/batch` | [异步 RPC 用户函数示例](distributed/rpc/batch/README.md) |
| `distributed/rpc/ddp_rpc` | [结合分布式数据并行（DDP）与分布式 RPC 的示例](distributed/rpc/ddp_rpc/README.md) |
| `distributed/rpc/parameter_server` | [基于 RPC 的分布式训练](distributed/rpc/parameter_server/README.md) |
| `distributed/rpc/pipeline` | [分布式流水线并行示例](distributed/rpc/pipeline/README.md) |
| `distributed/rpc/rl` | [多观察者、单智能体的分布式强化学习示例](distributed/rpc/rl/README.md) |
| `distributed/rpc/rnn` | [分布式 RNN 模型并行示例](distributed/rpc/rnn/README.md) |
| `distributed/tensor_parallelism` | [使用 PyTorch 原生张量并行进行分布式训练](distributed/tensor_parallelism/README.md) |
| `fast_neural_style` | [快速神经风格迁移 :city_sunrise: :rocket:](fast_neural_style/README.md) |
| `fx/native_interpreter` | [使用 FX 和 TorchScript 自定义类，将 PyTorch 代码转换到原生运行时](fx/native_interpreter/README.md) |
| `fx` | [FX 示例](fx/README.md) |
| `gat` | [图注意力网络（GAT）](gat/README.md) |
| `gcn` | [图卷积网络（GCN）](gcn/README.md) |
| `imagenet` | [使用 PyTorch 在 ImageNet 上训练](imagenet/README.md) |
| `language_translation` | [语言翻译](language_translation/README.md) |
| `legacy/snli` | [基于 PyTorch 和 SNLI 的自然语言推理训练](legacy/snli/README.md) |
| `mnist` | [MNIST 基础示例](mnist/README.md) |
| `mnist_forward_forward` | [Forward-Forward 基础示例](mnist_forward_forward/README.md) |
| `mnist_hogwild` | [MNIST Hogwild 多进程训练示例](mnist_hogwild/README.md) |
| `mnist_rnn` | [使用 RNN 识别 MNIST 手写数字](mnist_rnn/README.md) |
| `regression` | [线性回归示例](regression/README.md) |
| `reinforcement_learning` | [强化学习训练示例](reinforcement_learning/README.md) |
| `siamese_network` | [孪生网络（Siamese Network）示例](siamese_network/README.md) |
| `super_resolution` | [使用高效亚像素卷积神经网络实现超分辨率](super_resolution/README.md) |
| `time_sequence_prediction` | [时间序列预测](time_sequence_prediction/README.md) |
| `vae` | [变分自编码器（VAE）基础示例](vae/README.md) |
| `word_language_model` | [使用 RNN 和 Transformer 进行词级语言建模](word_language_model/README.md) |
