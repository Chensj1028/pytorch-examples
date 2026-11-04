# 多观察者、单智能体的分布式强化学习示例

[English original](README.md)

> 中文说明；代码块中的命令、代码及程序输出保留原样。

本示例通过 CartPole 强化学习任务演示 `torch.distributed.rpc` API。重点是展示 RPC API 的用法，并非构建性能最好的 CartPole 求解器。

```
pip install -r requirements.txt
python main.py
```
