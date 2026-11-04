# PyTorch 分布式训练教程系列

[English original](README.md)

> 中文说明；代码块中的命令、代码及程序输出保留原样。

本目录是 [DDP 教程系列](https://pytorch.org/tutorials/beginner/ddp_series_intro.html) 的配套代码。

每个代码文件都在前一个文件的基础上扩展。教程从单 GPU、非分布式训练脚本开始，逐步演进到 Slurm 集群上的多节点训练。

## 文件说明

- [single_gpu.py](single_gpu.py)：非分布式训练脚本。
- [multigpu.py](multigpu.py)：单节点 DDP 训练。
- [multigpu_torchrun.py](multigpu_torchrun.py)：使用 Torchrun 的单节点 DDP 训练。
- [multinode.py](multinode.py)：使用 Torchrun 进行多节点 DDP 训练，可选配 Slurm。
  - [slurm/setup_pcluster_slurm.md](slurm/setup_pcluster_slurm.md)：配置 AWS 集群的说明。
  - [slurm/config.yaml.template](slurm/config.yaml.template)：AWS 集群配置模板。
  - [slurm/sbatch_run.sh](slurm/sbatch_run.sh)：启动训练任务的 Slurm 脚本。
