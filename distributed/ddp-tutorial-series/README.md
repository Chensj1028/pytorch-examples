# distributed-pytorch

PyTorch 分布式训练教程系列

Code for the DDP tutorial series at https://pytorch.org/tutorials/beginner/ddp_series_intro.html

本目录是 [DDP 教程系列](https://pytorch.org/tutorials/beginner/ddp_series_intro.html) 的配套代码。

Each code file extends upon the previous one. The series starts with a non-distributed script that runs on a single GPU and incrementally updates to end with multinode training on a Slurm cluster.

每个代码文件都在前一个文件的基础上扩展。教程从单 GPU、非分布式训练脚本开始，逐步演进到 Slurm 集群上的多节点训练。

## Files

文件说明

* [single_gpu.py](single_gpu.py): Non-distributed training script

  [single_gpu.py](single_gpu.py)：非分布式训练脚本。

* [multigpu.py](multigpu.py): DDP on a single node

  [multigpu.py](multigpu.py)：单节点 DDP 训练。

* [multigpu_torchrun.py](multigpu_torchrun.py): DDP on a single node using Torchrun

  [multigpu_torchrun.py](multigpu_torchrun.py)：使用 Torchrun 的单节点 DDP 训练。

* [multinode.py](multinode.py): DDP on multiple nodes using Torchrun (and optionally Slurm)

  [multinode.py](multinode.py)：使用 Torchrun 进行多节点 DDP 训练，可选配 Slurm。

    * [slurm/setup_pcluster_slurm.md](slurm/setup_pcluster_slurm.md): instructions to set up an AWS cluster

      [slurm/setup_pcluster_slurm.md](slurm/setup_pcluster_slurm.md)：配置 AWS 集群的说明。

    * [slurm/config.yaml.template](slurm/config.yaml.template): configuration to set up an AWS cluster

      [slurm/config.yaml.template](slurm/config.yaml.template)：AWS 集群配置模板。

    * [slurm/sbatch_run.sh](slurm/sbatch_run.sh): slurm script to launch the training job

      [slurm/sbatch_run.sh](slurm/sbatch_run.sh)：启动训练任务的 Slurm 脚本。
