Distributed Pipeline Parallel Example

# 分布式流水线并行示例

This example shows how to distribute a ResNet50 model on two RPC workers and
then implement distributed pipeline parallelism using RPC. With pipeline
parallelism, every input batch is divided into micro-batches and thse
micro-batches are feed into the model in a pipelined fashion to increase the
amortized device utilization. Note that this example only parallelizes the
forward pass which can be viewed as the distributed counterpart of the
[single machine pipeline parallel](https://pytorch.org/tutorials/intermediate/model_parallel_tutorial.html#speed-up-by-pipelining-inputs)
example.

本示例将 ResNet50 模型分布到两个 RPC 工作进程上，并通过 RPC 实现分布式流水线并行。

流水线并行将每个输入批次拆成更小的微批次，再按流水线方式送入模型，以提高整体设备利用率。

本例只对前向传播进行并行处理，可以看作[单机流水线并行示例](https://pytorch.org/tutorials/intermediate/model_parallel_tutorial.html#speed-up-by-pipelining-inputs)的分布式版本。

```
pip install -r requirements.txt
python main.py
```
