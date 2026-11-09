# Custom Dataset Example with the PyTorch C++ Frontend

使用 PyTorch C++ 前端加载自定义数据集

This folder contains an example of loading a custom image dataset with OpenCV and training a model to label images, using the PyTorch C++ frontend.

本目录展示如何使用 OpenCV 加载自定义图片数据集，并通过 PyTorch C++ 前端训练图片分类模型。

The dataset used here is [Caltech 101](https://data.caltech.edu/records/mzrjq-6wc02) dataset.

本例使用 [Caltech 101 数据集](https://data.caltech.edu/records/mzrjq-6wc02)。

The entire training code is contained in custom-data.cpp.

完整训练代码位于 `custom-data.cpp`。

You can find instructions on how to install OpenCV [here](../tools/InstallingOpenCV.md).

OpenCV 安装说明见[这里](../tools/InstallingOpenCV.md)。

To build the code, run the following commands from your terminal:

在终端执行以下命令构建代码：

```shell
$ cd custom-dataset
$ mkdir build
$ cd build
$ cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
$ make
```

where /path/to/libtorch should be the path to the unzipped LibTorch distribution, which you can get from the [PyTorch homepage](https://pytorch.org/get-started/locally/).

其中 `/path/to/libtorch` 应替换为解压后的 LibTorch 目录。LibTorch 可从 [PyTorch 官网](https://pytorch.org/get-started/locally/)获取。

If you see an error like `undefined reference to cv::imread(std::string const&, int)` when running the `make` command, you should build LibTorch from source using the instructions [here](https://github.com/pytorch/pytorch#from-source), and then set `CMAKE_PREFIX_PATH` to that PyTorch source directory. An alternative solution is to use `libtorch-cxx11-abi-shared-with-deps` instead of `libtorch-shared-with-deps` as the latter is not compatible with openCV (reported [here](https://discuss.pytorch.org/t/library-conflict-between-libtorch-and-opencv/64489)).

如果执行 `make` 时遇到 `undefined reference to cv::imread(std::string const&, int)` 一类错误，可按照[源码构建说明](https://github.com/pytorch/pytorch#from-source)编译 LibTorch，然后将 `CMAKE_PREFIX_PATH` 设为该 PyTorch 源码目录。另一种方法是使用 `libtorch-cxx11-abi-shared-with-deps`，替代与 OpenCV 不兼容的 `libtorch-shared-with-deps`，相关讨论见[这里](https://discuss.pytorch.org/t/library-conflict-between-libtorch-and-opencv/64489)。

The build directory should look like this:

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

`info.txt` file gets copied from source directory during build.

构建过程中，`info.txt` 会从源码目录复制到构建目录。

Execute the compiled binary to train the model:

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
