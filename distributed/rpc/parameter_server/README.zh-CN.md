# 基于 RPC 的分布式训练

[English original](README.md)

> 中文说明；代码块中的命令、代码及程序输出保留原样。

这是一个基础示例，多个训练进程通过远程调用，共同训练托管在服务器上的模型。

本地运行时，为服务器和每个工作进程分别打开终端，执行 `python rpc_parameter_server.py --world_size=WORLD_SIZE --rank=RANK`。

例如，进程总数为 2 时，在一个终端运行 `python rpc_parameter_server.py --world_size=2 --rank=0` 启动主进程；在另一个终端运行 `python rpc_parameter_server.py --world_size=2 --rank=1` 启动训练进程，即可用一个服务器和一个训练进程开始训练。

为便于演示，本示例仅支持 0～2 个 GPU，但这种模式可以扩展到更多 GPU。训练命令中通过 `--num_gpus=N` 指定 GPU 数量。

原版列出的地址参数为 `--master_addr=<address>` 和 `master_port=PORT`，用于指定主工作进程监听的地址及端口。所有工作进程在相互发现、会合时都会联系主进程。默认 `master_addr` 为 `localhost`，`master_port` 为 29500。
