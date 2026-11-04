# C++ 自动求导示例

[English original](README.md)

> 中文说明；代码块中的命令、代码及程序输出保留原样。

`autograd.cpp` 包含多个使用 PyTorch C++ 前端进行自动求导的示例。

在终端执行以下命令构建代码：

```shell
$ cd autograd
$ mkdir build
$ cd build
$ cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
$ make
```

其中 `/path/to/libtorch` 应替换为解压后的 LibTorch 目录。LibTorch 可从 [PyTorch 官网](https://pytorch.org/get-started/locally/)获取。

运行编译生成的可执行文件：

```shell
$ ./autograd
====== Running: "Basic autograd operations" ======
 1  1
 1  1
[ CPUFloatType{2,2} ]
 3  3
 3  3
[ CPUFloatType{2,2} ]
AddBackward1
 27  27
 27  27
[ CPUFloatType{2,2} ]
MulBackward1
27
[ CPUFloatType{} ]
MeanBackward0
false
true
SumBackward0
 4.5000  4.5000
 4.5000  4.5000
[ CPUFloatType{2,2} ]
  813.6625
 1015.0142
 -664.8849
[ CPUFloatType{3} ]
MulBackward1
  204.8000
 2048.0000
    0.2048
[ CPUFloatType{3} ]
true
true
false
true
false
true

====== Running "Computing higher-order gradients in C++" ======
 0.0025  0.0946  0.1474  0.1387
 0.0238 -0.0018  0.0259  0.0094
 0.0513 -0.0549 -0.0604  0.0210
[ CPUFloatType{3,4} ]

====== Running "Using custom autograd function in C++" ======
-3.5513  3.7160  3.6477
-3.5513  3.7160  3.6477
[ CPUFloatType{2,3} ]
 0.3095  1.4035 -0.0349
 0.3095  1.4035 -0.0349
 0.3095  1.4035 -0.0349
 0.3095  1.4035 -0.0349
[ CPUFloatType{4,3} ]
 5.5000
 5.5000
[ CPUFloatType{2} ]
```
