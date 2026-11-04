# 使用 PyTorch C++ 前端训练 DCGAN

[English original](README.md)

> 中文说明；代码块中的命令、代码及程序输出保留原样。

本目录展示如何通过 PyTorch C++ 前端训练 DCGAN，生成 MNIST 手写数字图片。

完整训练代码位于 `dcgan.cpp`。

安装 argparse 的命令见[这里](https://github.com/pytorch/examples/blob/main/.github/workflows/main_cpp.yml#L34)。

在终端执行以下命令构建代码：

```shell
$ cd dcgan
$ mkdir build
$ cd build
$ cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
$ make
```

其中 `/path/to/libtorch` 应替换为解压后的 LibTorch 目录。LibTorch 可从 [PyTorch 官网](https://pytorch.org/get-started/locally/)获取。

运行编译生成的可执行文件来训练模型：

```shell
$ ./dcgan
[ 1/30][200/938] D_loss: 0.4953 | G_loss: 4.0195
-> checkpoint 1
[ 1/30][400/938] D_loss: 0.3610 | G_loss: 4.8148
-> checkpoint 2
[ 1/30][600/938] D_loss: 0.4072 | G_loss: 4.36760
-> checkpoint 3
[ 1/30][800/938] D_loss: 0.4444 | G_loss: 4.0250
-> checkpoint 4
[ 2/30][200/938] D_loss: 0.3761 | G_loss: 3.8790
-> checkpoint 5
[ 2/30][400/938] D_loss: 0.3977 | G_loss: 3.3315
-> checkpoint 6
[ 2/30][600/938] D_loss: 0.3815 | G_loss: 3.5696
-> checkpoint 7
[ 2/30][800/938] D_loss: 0.4039 | G_loss: 3.2759
-> checkpoint 8
[ 3/30][200/938] D_loss: 0.4236 | G_loss: 4.5132
-> checkpoint 9
[ 3/30][400/938] D_loss: 0.3645 | G_loss: 3.9759
-> checkpoint 10
...
```

通过 `--epochs` 指定训练轮数：

```shell
$ ./dcgan --epochs 10
```

未指定 `--epochs` 时，默认训练 30 轮。

训练脚本会定期生成图片样本。使用本目录的 `display_samples.py` 将样本绘制成图片，例如：

```shell
$ python display_samples.py -i dcgan-sample-10.pt
Saved out.png
```
