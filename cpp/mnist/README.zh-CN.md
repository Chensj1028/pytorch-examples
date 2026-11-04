# 使用 PyTorch C++ 前端识别 MNIST 手写数字

[English original](README.md)

> 中文说明；代码块中的命令、代码及程序输出保留原样。

本目录展示如何通过 PyTorch C++ 前端训练计算机视觉模型，识别 MNIST 图片中的手写数字。

完整训练代码位于 `mnist.cpp`。

在终端执行以下命令构建代码：

```shell
$ cd mnist
$ mkdir build
$ cd build
$ cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
$ make
```

其中 `/path/to/libtorch` 应替换为解压后的 LibTorch 目录。LibTorch 可从 [PyTorch 官网](https://pytorch.org/get-started/locally/)获取。

运行编译生成的可执行文件来训练模型：

```shell
$ ./mnist
Train Epoch: 1 [59584/60000] Loss: 0.4232
Test set: Average loss: 0.1989 | Accuracy: 0.940
Train Epoch: 2 [59584/60000] Loss: 0.1926
Test set: Average loss: 0.1338 | Accuracy: 0.959
Train Epoch: 3 [59584/60000] Loss: 0.1390
Test set: Average loss: 0.0997 | Accuracy: 0.969
Train Epoch: 4 [59584/60000] Loss: 0.1239
Test set: Average loss: 0.0875 | Accuracy: 0.972
...
```
