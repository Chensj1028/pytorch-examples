# 异步 RPC 用户函数示例

[English original](README.md)

> 中文说明；代码块中的命令、代码及程序输出保留原样。

本目录包含两个使用 [`@rpc.functions.async_execution`](https://pytorch.org/docs/master/rpc.html#torch.distributed.rpc.functions.async_execution) 的示例。

1. **同步批量更新参数服务器**：通过 `@rpc.functions.async_execution` 更新和获取参数，可作为批处理 RPC 的入门示例。

   ```
   pip install -r requirements.txt
   python parameter_server.py
   ```

2. **多个观察者与批处理智能体**：通过 `@rpc.functions.async_execution`，将多个观察到的状态批量送入策略网络，获取对应动作。

   ```
   pip install -r requirements.txt
   python reinforce.py
   ```
