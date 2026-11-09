# Distributed Data Parallel (DDP) Applications with PyTorch

使用 PyTorch 构建分布式数据并行（DDP）应用

This guide demonstrates how to structure a distributed model training application for convenient multi-node launches using `torchrun`.

本指南展示如何组织分布式模型训练应用，便于使用 `torchrun` 在多个节点上启动训练。

---

## Prerequisites

预备知识

You should be familiar with:

建议先了解：

- [PyTorch basics](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)

  [PyTorch 基础](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)

- [Writing distributed applications](https://pytorch.org/tutorials/intermediate/dist_tuto.html)

  [编写分布式应用](https://pytorch.org/tutorials/intermediate/dist_tuto.html)

- [Distributed model training](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)

  [分布式模型训练](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)

This tutorial uses the [`torch.nn.parallel.DistributedDataParallel`](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel) (DDP) class for data parallel training: multiple workers train the same global model on different data shards, compute local gradients, and synchronize them using AllReduce. In High-Performance Computing (HCP), this is called _Single Program Multiple Data_ (SPMD).

本教程使用 [`torch.nn.parallel.DistributedDataParallel`](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel)（DDP）进行数据并行训练：多个工作进程分别处理不同的数据分片，计算本地梯度，再通过 AllReduce 同步梯度，共同训练一个全局模型。在高性能计算中，这种方式称为**单程序多数据（SPMD）**。

---

## Application Process Topologies

应用的进程拓扑

A Distributed Data Parallel (DDP) application can be executed on
multiple nodes where each node can consist of multiple GPU
devices. Each node in turn can run multiple copies of the DDP
application, each of which processes its models on multiple GPUs.

DDP 应用可以在多个节点上执行，每个节点可以包含多个 GPU，也可以运行多个 DDP 应用进程，每个进程可使用多个 GPU 处理模型。

Let:

符号说明：

- _N_ = number of nodes

  _N_：节点数。

- _G_ = number of GPUs per node

  _G_：每个节点的 GPU 数量。

- _W_ = **World Size** = total number of processes

  _W_：**World Size**，所有进程的总数。

- _L_ = **Local World Size** = processes per node

  _L_：**Local World Size**，单个节点上的进程数。

Each process has:

每个进程具有：

- **Local rank**: in `[0, L-1]`

  **Local rank**：本节点内的进程编号，范围为 `[0, L-1]`。

- **Global rank**: in `[0, W-1]`

  **Global rank**：全局进程编号，范围为 `[0, W-1]`。

**Example:**
If you launch a DDP app on 2 nodes, each with 4 GPUs, and want each process to span 2 GPUs, the mapping is as follows:

**示例：**在 2 个节点上启动应用，每个节点有 4 个 GPU，每个进程使用 2 个 GPU，映射关系如下：

![ProcessMapping](https://user-images.githubusercontent.com/875518/77676984-4c81e400-6f4c-11ea-87d8-f2ff505a99da.png)

![进程与 GPU 的映射](https://user-images.githubusercontent.com/875518/77676984-4c81e400-6f4c-11ea-87d8-f2ff505a99da.png)

While there are quite a few ways to map processes to nodes, a good rule of thumb is to have one process span a single GPU. This enables the DDP application to have as many parallel reader streams as there are GPUs and in practice provides a good balance between I/O and computational costs. In the rest of this tutorial, we assume that the application follows this heuristic.

进程与节点有多种映射方式，通常建议每个进程只使用一个 GPU。这样可以让并行数据读取流的数量与 GPU 数量一致，在实践中较好地平衡 I/O 与计算开销。下文采用这一配置。

# Preparing and launching a DDP application

准备和启动 DDP 应用

Independent of how a DDP application is launched, each process needs a mechanism to know its global and local ranks. Once this is known, all processes create a `ProcessGroup` that enables them to participate in collective communication operations such as AllReduce.

无论采用何种启动方式，每个进程都需要知道自己的全局编号和本地编号。随后，所有进程创建 `ProcessGroup`，参与 AllReduce 等集合通信操作。

A convenient way to start multiple DDP processes and initialize all values needed to create a `ProcessGroup` is to use the [`torchrun`](https://docs.pytorch.org/docs/stable/elastic/run.html) script provided with PyTorch.

使用 PyTorch 提供的 [`torchrun`](https://docs.pytorch.org/docs/stable/elastic/run.html)，可以方便地启动多个 DDP 进程，并初始化创建 `ProcessGroup` 所需的配置。

---

## Sample Application

示例应用

This example is based on the ["Hello, World" DDP tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html).

本示例基于 [DDP 入门教程](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)。

The application calls the `spmd_main` entrypoint:

应用调用 `spmd_main` 入口函数：

```python
if __name__ == "__main__":
    spmd_main()
```

In `spmd_main`, the process group is initialized using the Accelerator API. The rest of the rendezvous information comes from environment variables set by `torchrun`:

在 `spmd_main` 中，通过 Accelerator API 初始化进程组，其余会合信息由 `torchrun` 设置的环境变量提供：

```python
def spmd_main():
    # These are the parameters used to initialize the process group
    # 这些参数用于初始化进程组。
    env_dict = {
        key: os.environ[key]
        for key in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE")
    }
    rank = int(env_dict['RANK'])
    local_rank = int(env_dict['LOCAL_RANK'])
    local_world_size = int(env_dict['LOCAL_WORLD_SIZE'])

    print(f"[{os.getpid()}] Initializing process group with: {env_dict}")
    acc = torch.accelerator.current_accelerator()
    vendor_backend = torch.distributed.get_default_backend_for_device(acc)
    torch.accelerator.set_device_index(rank)
    torch.distributed.init_process_group(backend=vendor_backend)

    demo_basic(rank)

    # Tear down the process group
    # 销毁进程组。
    torch.distributed.destroy_process_group()
```

**Key points:**

**要点：**

- Each process reads its rank and world size from environment variables.

  每个进程从环境变量读取自身编号和进程总数。

- The process group is initialized for distributed communication.

  初始化进程组，以进行分布式通信。

The training function, `demo_basic`, initializes the DDP model on the appropriate GPU:

训练函数 `demo_basic` 在对应 GPU 上初始化 DDP 模型：

```python
def demo_basic(rank):
    print(
        f"[{os.getpid()}] rank = {torch.distributed.get_rank()}, "
        + f"world_size = {torch.distributed.get_world_size()}"
    )

    model = ToyModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    optimizer.zero_grad()
    outputs = ddp_model(torch.randn(20, 10))
    labels = torch.randn(20, 5).to(rank)
    loss_fn(outputs, labels).backward()
    optimizer.step()
```

---

## Launching the Application

启动应用

```sh
torchrun --nnodes=1 --nproc_per_node=8 example.py
```

---

## Example Output

输出示例

Expected output:

预期输出如下：

```sh
*****************************************
Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed.
*****************************************
[238627] Initializing process group with: {'MASTER_ADDR': '127.0.0.1', 'MASTER_PORT': '29500', 'RANK': '0', 'WORLD_SIZE': '8'}
[238630] Initializing process group with: {'MASTER_ADDR': '127.0.0.1', 'MASTER_PORT': '29500', 'RANK': '3', 'WORLD_SIZE': '8'}
[238628] Initializing process group with: {'MASTER_ADDR': '127.0.0.1', 'MASTER_PORT': '29500', 'RANK': '1', 'WORLD_SIZE': '8'}
[238634] Initializing process group with: {'MASTER_ADDR': '127.0.0.1', 'MASTER_PORT': '29500', 'RANK': '7', 'WORLD_SIZE': '8'}
[238631] Initializing process group with: {'MASTER_ADDR': '127.0.0.1', 'MASTER_PORT': '29500', 'RANK': '4', 'WORLD_SIZE': '8'}
[238632] Initializing process group with: {'MASTER_ADDR': '127.0.0.1', 'MASTER_PORT': '29500', 'RANK': '5', 'WORLD_SIZE': '8'}
[238629] Initializing process group with: {'MASTER_ADDR': '127.0.0.1', 'MASTER_PORT': '29500', 'RANK': '2', 'WORLD_SIZE': '8'}
[238633] Initializing process group with: {'MASTER_ADDR': '127.0.0.1', 'MASTER_PORT': '29500', 'RANK': '6', 'WORLD_SIZE': '8'}
[238633] world_size = 8, rank = 6, backend=nccl
[238628] world_size = 8, rank = 1, backend=nccl
[238629] world_size = 8, rank = 2, backend=nccl
[238631] world_size = 8, rank = 4, backend=nccl
[238630] world_size = 8, rank = 3, backend=nccl
[238632] world_size = 8, rank = 5, backend=nccl
[238634] world_size = 8, rank = 7, backend=nccl
[238627] world_size = 8, rank = 0, backend=nccl
[238633] rank = 6, world_size = 8
[238628] rank = 1, world_size = 8
[238632] rank = 5, world_size = 8
[238634] rank = 7, world_size = 8
[238629] rank = 2, world_size = 8
[238630] rank = 3, world_size = 8
[238631] rank = 4, world_size = 8
[238627] rank = 0, world_size = 8
```

# Conclusions

小结

As the author of a distributed data parallel application, your code needs to be aware of two types of resources: compute nodes and the GPUs within each node. The process of setting up bookkeeping to track how the set of GPUs is mapped to the processes of your application can be tedious and error-prone. We hope that by structuring your application as shown in this example and using `torchrun`, the mechanics of setting up distributed training can be significantly simplified.

编写分布式数据并行应用时，需要同时管理计算节点和各节点上的 GPU。手动维护 GPU 与应用进程之间的映射比较繁琐，也容易出错。按照本示例组织应用，并使用 `torchrun`，可以简化分布式训练的配置和启动过程。
