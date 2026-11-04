# PyTorch 示例集（中文版）

[English original](README.md)

> 中文说明；代码块中的命令、代码及程序输出保留原样。

项目主页：https://pytorch.org/examples/

`pytorch/examples` 展示 [PyTorch](https://github.com/pytorch/pytorch) 的使用示例。项目旨在提供经过整理、简短、依赖较少或无需额外依赖的高质量示例，覆盖不同任务，方便在自己的项目中参考使用。

- 教程：[pytorch/tutorials](https://github.com/pytorch/tutorials)。
- PyTorch 官网内容修改：[pytorch/pytorch.github.io](https://github.com/pytorch/pytorch.github.io)。
- 通用模型中心：[PyTorch Hub](https://pytorch.org/hub/) 或 [Hugging Face 模型库](https://huggingface.co/models)。
- 生产环境中的 PyTorch 使用方案：[facebookresearch/recipes](https://github.com/facebookresearch/recipes)。
- 问答与支持：[PyTorch 论坛](https://discuss.pytorch.org/)。

## 初学者从这里开始

建议先阅读 [MNIST 手写数字识别](./mnist/README.zh-CN.md)，再结合本地的[入门说明](./入门说明.md)和 `mnist/main.py` 学习数据加载、模型、训练及测试。

## 示例与模型

- [使用卷积网络进行 MNIST 图片分类](./mnist/README.zh-CN.md)
- [使用 RNN 和 Transformer 进行词级语言建模](./word_language_model/README.zh-CN.md)
- [使用常见网络训练 ImageNet 分类器](./imagenet/README.zh-CN.md)
- [深度卷积生成对抗网络（DCGAN）](./dcgan/README.zh-CN.md)
- [变分自编码器（VAE）](./vae/README.zh-CN.md)
- [使用高效亚像素卷积网络实现超分辨率](./super_resolution/README.zh-CN.md)
- [使用 Hogwild 在多个进程间共享卷积网络，训练 MNIST 模型](mnist_hogwild/README.zh-CN.md)
- [使用 Actor-Critic 训练 CartPole 保持平衡](./reinforcement_learning/README.zh-CN.md)
- [结合 GloVe 词向量、LSTM 和 torchtext 进行 SNLI 自然语言推理](legacy/snli/README.zh-CN.md)
- [使用 LSTM 学习正弦波，预测时间序列](./time_sequence_prediction/README.zh-CN.md)
- [对图片进行神经风格迁移](./fast_neural_style/README.zh-CN.md)
- [在 OpenAI Gym 中使用 Actor-Critic 和 REINFORCE 进行强化学习](./reinforcement_learning/README.zh-CN.md)
- [使用 FX 进行 PyTorch 模块变换](./fx/README.zh-CN.md)
- 分布式 PyTorch 示例：[分布式数据并行（DDP）](./distributed/ddp/README.zh-CN.md)和 [RPC](./distributed/rpc)。
- [PyTorch C++ 前端示例](cpp)
- [使用 Forward-Forward 进行图片分类](./mnist_forward_forward/README.zh-CN.md)
- [使用 Transformer 进行语言翻译](./language_translation/README.zh-CN.md)

其他独立仓库中的示例：

- [使用带注意力机制的序列到序列 RNN 进行神经机器翻译（OpenNMT）](https://github.com/OpenNMT/OpenNMT-py)

## 参与贡献

如果想提交自己的示例或修复问题，请先阅读 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 完整中文文档索引

本仓库的 40 份 README 均已配套中文版。本页为首页，以下列出其余 39 份：

| 示例目录 | 中文说明 |
| --- | --- |
| `cpp/autograd` | [C++ 自动求导示例](cpp/autograd/README.zh-CN.md) |
| `cpp/custom-dataset` | [使用 PyTorch C++ 前端加载自定义数据集](cpp/custom-dataset/README.zh-CN.md) |
| `cpp/dcgan` | [使用 PyTorch C++ 前端训练 DCGAN](cpp/dcgan/README.zh-CN.md) |
| `cpp/distributed` | [使用 PyTorch C++ 前端（LibTorch）在 MNIST 上进行分布式训练](cpp/distributed/README.zh-CN.md) |
| `cpp/mnist` | [使用 PyTorch C++ 前端识别 MNIST 手写数字](cpp/mnist/README.zh-CN.md) |
| `cpp/regression` | [C++ 线性回归示例](cpp/regression/README.zh-CN.md) |
| `cpp/transfer-learning` | [使用 LibTorch 和 OpenCV 在猫狗数据集上进行迁移学习](cpp/transfer-learning/README.zh-CN.md) |
| `dcgan` | [深度卷积生成对抗网络（DCGAN）](dcgan/README.zh-CN.md) |
| `distributed/ddp` | [使用 PyTorch 构建分布式数据并行（DDP）应用](distributed/ddp/README.zh-CN.md) |
| `distributed/ddp-tutorial-series` | [PyTorch 分布式训练教程系列](distributed/ddp-tutorial-series/README.zh-CN.md) |
| `distributed/FSDP` | [FSDP1 T5 示例](distributed/FSDP/README.zh-CN.md) |
| `distributed/FSDP2` | [FSDP2 示例](distributed/FSDP2/README.zh-CN.md) |
| `distributed/minGPT-ddp` | [minGPT-DDP](distributed/minGPT-ddp/README.zh-CN.md) |
| `distributed/rpc/batch` | [异步 RPC 用户函数示例](distributed/rpc/batch/README.zh-CN.md) |
| `distributed/rpc/ddp_rpc` | [结合分布式数据并行（DDP）与分布式 RPC 的示例](distributed/rpc/ddp_rpc/README.zh-CN.md) |
| `distributed/rpc/parameter_server` | [基于 RPC 的分布式训练](distributed/rpc/parameter_server/README.zh-CN.md) |
| `distributed/rpc/pipeline` | [分布式流水线并行示例](distributed/rpc/pipeline/README.zh-CN.md) |
| `distributed/rpc/rl` | [多观察者、单智能体的分布式强化学习示例](distributed/rpc/rl/README.zh-CN.md) |
| `distributed/rpc/rnn` | [分布式 RNN 模型并行示例](distributed/rpc/rnn/README.zh-CN.md) |
| `distributed/tensor_parallelism` | [使用 PyTorch 原生张量并行进行分布式训练](distributed/tensor_parallelism/README.zh-CN.md) |
| `fast_neural_style` | [快速神经风格迁移 :city_sunrise: :rocket:](fast_neural_style/README.zh-CN.md) |
| `fx/native_interpreter` | [使用 FX 和 TorchScript 自定义类，将 PyTorch 代码转换到原生运行时](fx/native_interpreter/README.zh-CN.md) |
| `fx` | [FX 示例](fx/README.zh-CN.md) |
| `gat` | [图注意力网络（GAT）](gat/README.zh-CN.md) |
| `gcn` | [图卷积网络（GCN）](gcn/README.zh-CN.md) |
| `imagenet` | [使用 PyTorch 在 ImageNet 上训练](imagenet/README.zh-CN.md) |
| `language_translation` | [语言翻译](language_translation/README.zh-CN.md) |
| `legacy/snli` | [基于 PyTorch 和 SNLI 的自然语言推理训练](legacy/snli/README.zh-CN.md) |
| `mnist` | [MNIST 基础示例](mnist/README.zh-CN.md) |
| `mnist_forward_forward` | [Forward-Forward 基础示例](mnist_forward_forward/README.zh-CN.md) |
| `mnist_hogwild` | [MNIST Hogwild 多进程训练示例](mnist_hogwild/README.zh-CN.md) |
| `mnist_rnn` | [使用 RNN 识别 MNIST 手写数字](mnist_rnn/README.zh-CN.md) |
| `regression` | [线性回归示例](regression/README.zh-CN.md) |
| `reinforcement_learning` | [强化学习训练示例](reinforcement_learning/README.zh-CN.md) |
| `siamese_network` | [孪生网络（Siamese Network）示例](siamese_network/README.zh-CN.md) |
| `super_resolution` | [使用高效亚像素卷积神经网络实现超分辨率](super_resolution/README.zh-CN.md) |
| `time_sequence_prediction` | [时间序列预测](time_sequence_prediction/README.zh-CN.md) |
| `vae` | [变分自编码器（VAE）基础示例](vae/README.zh-CN.md) |
| `word_language_model` | [使用 RNN 和 Transformer 进行词级语言建模](word_language_model/README.zh-CN.md) |
