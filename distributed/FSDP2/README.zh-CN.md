# FSDP2 示例

[English original](README.md)

> 中文说明；代码块中的命令、代码及程序输出保留原样。

在 Transformer 模型上运行 FSDP2：

```
cd distributed/FSDP2
pip install -r requirements.txt
torchrun --nproc_per_node 2 example.py
```

- 第一次运行会创建 `checkpoints` 目录，将状态字典保存到该目录。
- 第二次运行会加载先前的检查点。

启用显式预取：

```
torchrun --nproc_per_node 2 example.py --explicit-prefetch
```

启用混合精度：

```
torchrun --nproc_per_node 2 example.py --mixed-precision
```

演示分布式检查点（DCP）API：

```
torchrun --nproc_per_node 2 example.py --dcp-api
```

## PyTorch 版本

原版要求至少使用 PyTorch 2.5，并建议使用当时较新的 nightly 构建。安装方式见 https://pytorch.org/get-started/locally/ 。
