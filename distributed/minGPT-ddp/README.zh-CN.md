# minGPT-DDP

[English original](README.md)

> 中文说明；代码块中的命令、代码及程序输出保留原样。

本目录为[教程](https://pytorch.org/tutorials/intermediate/ddp_series_minGPT.html)提供配套代码，展示如何在 PyTorch 中通过分布式数据并行（DDP）训练类似 GPT 的模型。

带星号（*）的文件改编自 [minGPT 仓库](https://github.com/karpathy/minGPT)。

- [trainer.py](mingpt/trainer.py)：包含 `Trainer` 类，使用给定数据集执行模型的分布式训练迭代。
- [model.py *](mingpt/model.py)：定义模型结构。
- [char_dataset.py *](mingpt/char_dataset.py)：包含字符级数据集的 `Dataset` 类。
- [gpt2_train_cfg.yaml](mingpt/gpt2_train_cfg.yaml)：包含数据、模型、优化器和训练配置。
- [main.py](mingpt/main.py)：训练任务入口，负责初始化 DDP 进程组、读取配置并运行训练。
- [slurm/](mingpt/slurm)：包含配置 AWS 集群，以及运行多节点训练的 Slurm 脚本等文件。
