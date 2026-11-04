# C++ 线性回归示例

[English original](README.md)

> 中文说明；代码块中的命令、代码及程序输出保留原样。

训练单个全连接层，拟合一个四次多项式。

在终端执行以下命令构建代码：

```shell
$ cd regression
$ mkdir build
$ cd build
$ cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
$ make
```

其中 `/path/to/libtorch` 应替换为解压后的 LibTorch 目录。LibTorch 可从 [PyTorch 官网](https://pytorch.org/get-started/locally/)获取。

运行编译生成的可执行文件：

```shell
$ ./regression
Loss: 0.000301158 after 584 batches
==> Learned function:	y = 11.6441 x^4 -3.10164 x^3 2.19786 x^2 -3.83606 x^1 + 4.37066
==> Actual function:	y = 11.669 x^4 -3.16023 x^3 2.19182 x^2 -3.81505 x^1 + 4.38219
...
```
