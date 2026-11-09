Distributed Multi-Observer Single-Agent Reinforcement Learning Example

# 多观察者、单智能体的分布式强化学习示例

This example demonstrates `torch.distributed.rpc` API using an CartPole
reinforcement learning example. Please note that the goal is to present the RPC
API instead of building the best CartPole solver.

本示例通过 CartPole 强化学习任务演示 `torch.distributed.rpc` API。重点是展示 RPC API 的用法，并非构建性能最好的 CartPole 求解器。

```
pip install -r requirements.txt
python main.py
```
