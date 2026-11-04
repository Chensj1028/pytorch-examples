# 深度卷积生成对抗网络（DCGAN）

[English original](README.md)

> 中文说明；代码块中的命令、代码及程序输出保留原样。

本示例实现论文 [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](http://arxiv.org/abs/1511.06434)。

实现方式与 Torch 版本 [dcgan.torch](https://github.com/soumith/dcgan.torch) 非常接近。

每训练 100 次迭代，程序会将真实样本和生成模型的样本写入 `real_samples.png`、`fake_samples.png`。

每轮训练结束后，模型保存为 `netG_epoch_%d.pth` 和 `netD_epoch_%d.pth`。

## 下载数据集

可以克隆 [LSUN 仓库](https://github.com/fyu/lsun)，然后运行以下命令下载 LSUN 数据集：

```
python download.py -c bedroom
```

## 使用方法

```
usage: main.py [-h] --dataset DATASET [--dataroot DATAROOT]
               [--workers WORKERS] [--batchSize BATCHSIZE]
               [--imageSize IMAGESIZE] [--nz NZ] [--ngf NGF] [--ndf NDF]
               [--niter NITER] [--lr LR] [--beta1 BETA1] [--dry-run]
               [--ngpu NGPU] [--netG NETG] [--netD NETD] [--outf OUTF]
               [--manualSeed MANUALSEED] [--classes CLASSES] [--accel]

options:
  -h, --help            show this help message and exit
  --dataset DATASET     cifar10 | lsun | mnist |imagenet | folder | lfw | fake
  --dataroot DATAROOT   path to dataset
  --workers WORKERS     number of data loading workers
  --batchSize BATCHSIZE
                        input batch size
  --imageSize IMAGESIZE
                        the height / width of the input image to network
  --nz NZ               size of the latent z vector
  --ngf NGF             number of generator filters
  --ndf NDF             number of discriminator filters
  --niter NITER         number of epochs to train for
  --lr LR               learning rate, default=0.0002
  --beta1 BETA1         beta1 for adam. default=0.5
  --dry-run             check a single training cycle works
  --ngpu NGPU           number of GPUs to use
  --netG NETG           path to netG (to continue training)
  --netD NETD           path to netD (to continue training)
  --outf OUTF           folder to output images and model checkpoints
  --manualSeed MANUALSEED
                        manual seed
  --classes CLASSES     comma separated list of classes for the lsun data set
  --accel               enables accelerator
```
