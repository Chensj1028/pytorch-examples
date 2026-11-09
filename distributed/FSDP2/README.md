## FSDP2

FSDP2 示例

To run FSDP2 on transformer model:

在 Transformer 模型上运行 FSDP2：

```
cd distributed/FSDP2
pip install -r requirements.txt
torchrun --nproc_per_node 2 example.py
```

* For 1st time, it creates a "checkpoints" folder and saves state dicts there

  第一次运行会创建 `checkpoints` 目录，将状态字典保存到该目录。

* For 2nd time, it loads from previous checkpoints

  第二次运行会加载先前的检查点。

To enable explicit prefetching

启用显式预取：

```
torchrun --nproc_per_node 2 example.py --explicit-prefetch
```

To enable mixed precision

启用混合精度：

```
torchrun --nproc_per_node 2 example.py --mixed-precision
```

To showcase DCP API

演示分布式检查点（DCP）API：

```
torchrun --nproc_per_node 2 example.py --dcp-api
```

## Ensure you are running a recent version of PyTorch:

PyTorch 版本

see https://pytorch.org/get-started/locally/ to install at least 2.5 and ideally a current nightly build.

原版要求至少使用 PyTorch 2.5，并建议使用当时较新的 nightly 构建。安装方式见 https://pytorch.org/get-started/locally/ 。
