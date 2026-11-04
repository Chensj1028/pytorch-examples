# 分布式流水线并行示例

[English original](README.md)

> 中文说明；代码块中的命令、代码及程序输出保留原样。

本示例将 ResNet50 模型分布到两个 RPC 工作进程上，并通过 RPC 实现分布式流水线并行。

流水线并行将每个输入批次拆成更小的微批次，再按流水线方式送入模型，以提高整体设备利用率。

本例只对前向传播进行并行处理，可以看作[单机流水线并行示例](https://pytorch.org/tutorials/intermediate/model_parallel_tutorial.html#speed-up-by-pipelining-inputs)的分布式版本。

```
pip install -r requirements.txt
python main.py
```
