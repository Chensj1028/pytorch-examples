# 使用 PyTorch C++ 前端加载自定义数据集

[English original](README.md)

> 中文说明；代码块中的命令、代码及程序输出保留原样。

本目录展示如何使用 OpenCV 加载自定义图片数据集，并通过 PyTorch C++ 前端训练图片分类模型。

本例使用 [Caltech 101 数据集](https://data.caltech.edu/records/mzrjq-6wc02)。

完整训练代码位于 `custom-data.cpp`。

OpenCV 安装说明见[这里](../tools/InstallingOpenCV.md)。

在终端执行以下命令构建代码：

```shell
$ cd custom-dataset
$ mkdir build
$ cd build
$ cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
$ make
```

其中 `/path/to/libtorch` 应替换为解压后的 LibTorch 目录。LibTorch 可从 [PyTorch 官网](https://pytorch.org/get-started/locally/)获取。

如果执行 `make` 时遇到 `undefined reference to cv::imread(std::string const&, int)` 一类错误，可按照[源码构建说明](https://github.com/pytorch/pytorch#from-source)编译 LibTorch，然后将 `CMAKE_PREFIX_PATH` 设为该 PyTorch 源码目录。另一种方法是使用 `libtorch-cxx11-abi-shared-with-deps`，替代与 OpenCV 不兼容的 `libtorch-shared-with-deps`，相关讨论见[这里](https://discuss.pytorch.org/t/library-conflict-between-libtorch-and-opencv/64489)。

构建后的目录结构如下：

```
.
├── custom-dataset
├── dataset
│   ├── accordion
│   │   ├── image_0001.jpg
│   │   ├── ...
│   ├── airplanes
│   │   ├── ...
│   ├── ...
├── info.txt
└── Makefile
└── ...
```

构建过程中，`info.txt` 会从源码目录复制到构建目录。

运行编译生成的可执行文件来训练模型：

```shell
./custom-dataset
Running on: CUDA
Train Epoch: 1 16/7281	Loss: 0.314655	Acc: 0
Train Epoch: 1 176/7281	Loss: 0.532111	Acc: 0.0681818
Train Epoch: 1 336/7281	Loss: 0.538482	Acc: 0.0714286
Train Epoch: 1 496/7281	Loss: 0.535302	Acc: 0.0705645
Train Epoch: 1 656/7281	Loss: 0.536113	Acc: 0.0716463
Train Epoch: 1 816/7281	Loss: 0.537626	Acc: 0.0784314
Train Epoch: 1 976/7281	Loss: 0.537055	Acc: 0.079918
...

```
