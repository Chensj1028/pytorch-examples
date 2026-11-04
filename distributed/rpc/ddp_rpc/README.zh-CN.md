# 结合分布式数据并行（DDP）与分布式 RPC 的示例

[English original](README.md)

> 中文说明；代码块中的命令、代码及程序输出保留原样。

本示例展示如何结合 DDP 与分布式 RPC 框架。运行需要两个各带一个 GPU 的训练节点、一个主节点和一个参数服务器。

主节点在参数服务器上初始化嵌入表，并协调各训练进程的训练循环。模型由两个部分组成：稠密部分 `nn.Linear` 通过 DDP 复制到各训练进程；稀疏部分 `nn.EmbeddingBag` 位于参数服务器。

每个训练进程通过 RPC 在参数服务器上查找嵌入向量，再将结果送入本地 `nn.Linear`。反向传播时，DDP 使用 AllReduce 聚合稠密部分的梯度，分布式反向传播则用于更新参数服务器上的嵌入表参数。

```
pip install -r requirements.txt
python main.py
```
