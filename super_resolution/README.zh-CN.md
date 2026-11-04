# 使用高效亚像素卷积神经网络实现超分辨率

[English original](README.md)

> 中文说明；代码块中的命令、代码及程序输出保留原样。

本示例展示如何使用 Shi 等人在论文 [Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network](https://arxiv.org/abs/1609.05158) 中提出的高效亚像素卷积层，提升网络中特征的空间分辨率，以完成超分辨率等任务。

```
usage: main.py [-h] --upscale_factor UPSCALE_FACTOR [--batchSize BATCHSIZE]
               [--testBatchSize TESTBATCHSIZE] [--nEpochs NEPOCHS] [--lr LR]
               [--accel] [--threads THREADS] [--seed SEED]

PyTorch Super Res Example

optional arguments:
  -h, --help            show this help message and exit
  --upscale_factor      super resolution upscale factor
  --batchSize           training batch size
  --testBatchSize       testing batch size
  --nEpochs             number of epochs to train for
  --lr                  Learning Rate. Default=0.01
  --accel                use accelerator
  --threads             number of threads for data loader to use Default=4
  --seed                random seed to use. Default=123
```

本示例使用 [BSD300 数据集](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/) 训练超分辨率网络：从 200 张训练图片中裁剪图像块进行训练，从 100 张测试图片中裁剪图像块进行评估。每轮训练结束后，保存名为 `model_epoch_<epoch_number>.pth` 的模型快照。

## 使用示例

### 训练

```bash
python main.py --upscale_factor 3 --batchSize 4 --testBatchSize 100 --nEpochs 30 --lr 0.001 --accel
```

### 对图片进行超分辨率处理

```bash
python super_resolve.py --input_image dataset/BSDS300/images/test/16077.jpg --model model_epoch_30.pth --output_filename out.png --accel
```
