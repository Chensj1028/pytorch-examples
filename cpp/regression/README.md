# Linear regression example

C++ 线性回归示例

Trains a single fully-connected layer to fit a 4th degree polynomial.

训练单个全连接层，拟合一个四次多项式。

To build the code, run the following commands from your terminal:

在终端执行以下命令构建代码：

```shell
$ cd regression
$ mkdir build
$ cd build
$ cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
$ make
```

where `/path/to/libtorch` should be the path to the unzipped _LibTorch_
distribution, which you can get from the [PyTorch
homepage](https://pytorch.org/get-started/locally/).

其中 `/path/to/libtorch` 应替换为解压后的 LibTorch 目录。LibTorch 可从 [PyTorch 官网](https://pytorch.org/get-started/locally/)获取。

Execute the compiled binary to run:

运行编译生成的可执行文件：

```shell
$ ./regression
Loss: 0.000301158 after 584 batches
==> Learned function:	y = 11.6441 x^4 -3.10164 x^3 2.19786 x^2 -3.83606 x^1 + 4.37066
==> Actual function:	y = 11.669 x^4 -3.16023 x^3 2.19182 x^2 -3.81505 x^1 + 4.38219
...
```
