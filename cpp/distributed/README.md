# Distributed Training on MNIST using PyTorch C++ Frontend (Libtorch)

使用 PyTorch C++ 前端（LibTorch）在 MNIST 上进行分布式训练

This folder contains an example of data-parallel training of a convolutional neural network on the MNIST dataset. For parallelization, Message Passing Interface (MPI) is used.

本目录展示如何在 MNIST 数据集上，对卷积神经网络进行数据并行训练。并行通信使用消息传递接口（MPI）。

The entire code is contained in dist-mnist.cpp

完整代码位于 `dist-mnist.cpp`。

You can find instructions on how to install MPI [here](https://www.open-mpi.org/faq/?category=building). This code was tested on Open MPI but it should run on other MPI distributions as well such as MPICH, MVAPICH, etc.

MPI 安装说明见 [Open MPI 文档](https://www.open-mpi.org/faq/?category=building)。本例已在 Open MPI 上测试，原则上也可用于 MPICH、MVAPICH 等其他 MPI 实现。

To build the code, run the following commands from the terminal:

在终端执行以下命令构建代码：

```shell
$ cd distributed
$ mkdir build
$ cd build
$ cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
$ make
```

where /path/to/libtorch should be the path to the unzipped LibTorch distribution. Note that the LibTorch from the [PyTorch homepage](https://pytorch.org/get-started/locally/) does not include MPI headers and cannot be used for this example. You have to compile LibTorch manually - a set of guidelines is provided [here](https://gist.github.com/lasagnaphil/3e0099816837318e8e8bcab7edcfd5d9), however this may vary for different systems.

其中 `/path/to/libtorch` 应替换为解压后的 LibTorch 目录。请注意，[PyTorch 官网](https://pytorch.org/get-started/locally/)提供的 LibTorch 不包含 MPI 头文件，不能直接用于本示例，需要手动编译 LibTorch。[这份说明](https://gist.github.com/lasagnaphil/3e0099816837318e8e8bcab7edcfd5d9)可供参考，实际步骤可能因系统而异。

To run the code,

运行代码：

```shell
mpirun -np {NUM-PROCS} ./dist-mnist
```
