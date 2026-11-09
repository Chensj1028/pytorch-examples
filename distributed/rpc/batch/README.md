# Examples For Asynchronous RPC User Functions

异步 RPC 用户函数示例

This folder contains two examples for [`@rpc.functions.async_execution`](https://pytorch.org/docs/master/rpc.html#torch.distributed.rpc.functions.async_execution):

本目录包含两个使用 [`@rpc.functions.async_execution`](https://pytorch.org/docs/master/rpc.html#torch.distributed.rpc.functions.async_execution) 的示例。

1. Synchronized Batch Update Parameter Server: uses `@rpc.functions.async_execution`
   for parameter update and retrieving. This serves as a simple starter example
   for batch RPC.

   **同步批量更新参数服务器**：通过 `@rpc.functions.async_execution` 更新和获取参数，可作为批处理 RPC 的入门示例。

   ```
   pip install -r requirements.txt
   python parameter_server.py
   ```

2. Multi-Observer with Batch-Processing Agent: uses `@rpc.functions.async_execution`
   to run multiple observed states through the policy to get actions.

   **多个观察者与批处理智能体**：通过 `@rpc.functions.async_execution`，将多个观察到的状态批量送入策略网络，获取对应动作。

   ```
   pip install -r requirements.txt
   python reinforce.py
   ```
