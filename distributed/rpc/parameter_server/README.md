### RPC-based distributed training

基于 RPC 的分布式训练

This is a basic example of RPC-based training that uses several trainers remotely train a model hosted on a server.

这是一个基础示例，多个训练进程通过远程调用，共同训练托管在服务器上的模型。

To run the example locally, run the following command worker for the server and each worker you wish to spawn, in separate terminal windows:
`python rpc_parameter_server.py --world_size=WORLD_SIZE --rank=RANK`. For example, for a master node with world size of 2, the command would be `python rpc_parameter_server.py --world_size=2 --rank=0`. The trainer can then be launched with the command `python rpc_parameter_server.py --world_size=2 --rank=1` in a separate window, and this will begin training with one server and a single trainer.

本地运行时，为服务器和每个工作进程分别打开终端，执行 `python rpc_parameter_server.py --world_size=WORLD_SIZE --rank=RANK`。

例如，进程总数为 2 时，在一个终端运行 `python rpc_parameter_server.py --world_size=2 --rank=0` 启动主进程；在另一个终端运行 `python rpc_parameter_server.py --world_size=2 --rank=1` 启动训练进程，即可用一个服务器和一个训练进程开始训练。

Note that for demonstration purposes, this example supports only between 0-2 GPUs, although the pattern can be extended to make use of additional GPUs. To configure the number of GPUs, pass in `--num_gpus=N` to your training command.

为便于演示，本示例仅支持 0～2 个 GPU，但这种模式可以扩展到更多 GPU。训练命令中通过 `--num_gpus=N` 指定 GPU 数量。

You can pass in the command line arguments `--master_addr=<address>` and `master_port=PORT` to indicate the address:port that the master worker is listening on. All workers will contact the master for rendezvous during worker discovery. By default, `master_addr` will be `localhost` and `master_port` will be 29500.

原版列出的地址参数为 `--master_addr=<address>` 和 `master_port=PORT`，用于指定主工作进程监听的地址及端口。所有工作进程在相互发现、会合时都会联系主进程。默认 `master_addr` 为 `localhost`，`master_port` 为 29500。
