# 孪生网络（Siamese Network）示例

[English original](README.md)

> 中文说明；代码块中的命令、代码及程序输出保留原样。

孪生网络用于估计图片相似度。网络包含两个相同的子网络，分别处理两张输入图片；将两个子网络的输出拼接后，送入线性层，再经过 sigmoid 函数得到结果。

[FaceNet](https://arxiv.org/pdf/1503.03832.pdf) 是孪生网络的一种变体。本实现使用论文 [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf) 中的 `ResNet-18` 作为特征提取器，因此与 FaceNet 有所不同。

另外，由于 MNIST 数据集较简单，本例使用 `BCELoss` 即可完成任务，没有使用 `TripletLoss`。

### 使用方法

安装依赖：

```bash
pip install -r requirements.txt
```

运行示例：

```bahs
python main.py
# CUDA_VISIBLE_DEVICES=2 python main.py  # to specify GPU id to ex. 2
```

检测到硬件加速设备时，示例会使用该设备；否则使用 CPU。

如需强制使用 CPU，添加 `--no-accel` 参数：

```bash
python main.py --no-accel
```

还可以添加以下参数，自定义运行方式：

```bash
--batch-size            input batch size for training (default: 64)
--test-batch-size       input batch size for testing (default: 1000)
--epochs                number of epochs to train (default: 14)
--lr                    learning rate (default: 1.0)
--gamma                 learning rate step gamma (default: 0.7)
--no-accel              disables accelerator
--dry-run               quickly check a single pass
--seed                  random seed (default: 1)
--log-interval          how many batches to wait before logging training status
--save-model            Saving the current Model
```
