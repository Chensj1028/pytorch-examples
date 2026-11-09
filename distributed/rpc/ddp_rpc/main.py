import random

import torch
import torch.distributed as dist
import torch.distributed.autograd as dist_autograd
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp
import torch.optim as optim
from torch.distributed.nn import RemoteModule
from torch.distributed.optim import DistributedOptimizer
from torch.distributed.rpc import RRef
from torch.distributed.rpc import TensorPipeRpcBackendOptions
from torch.nn.parallel import DistributedDataParallel as DDP

NUM_EMBEDDINGS = 100
EMBEDDING_DIM = 16

def verify_min_gpu_count(min_gpus: int = 2) -> bool:
    """ verification that we have at least 2 gpus to run dist examples """
    # 检查至少有两个 GPU，以运行分布式示例。
    has_gpu = torch.accelerator.is_available()
    gpu_count = torch.accelerator.device_count()
    return has_gpu and gpu_count >= min_gpus

class HybridModel(torch.nn.Module):
    r"""
    The model consists of a sparse part and a dense part.
    1) The dense part is an nn.Linear module that is replicated across all trainers using DistributedDataParallel.
    2) The sparse part is a Remote Module that holds an nn.EmbeddingBag on the parameter server.
    This remote model can get a Remote Reference to the embedding table on the parameter server.
    """
    # 模型包含稀疏部分和稠密部分：
    # 1）稠密部分为 nn.Linear，通过 DDP 复制到所有训练进程。
    # 2）稀疏部分为远程模块，在参数服务器上保存 nn.EmbeddingBag。
    # 可通过远程引用访问参数服务器上的嵌入表。

    def __init__(self, remote_emb_module, rank):
        super(HybridModel, self).__init__()
        self.remote_emb_module = remote_emb_module
        self.fc = DDP(torch.nn.Linear(16, 8).to(rank))
        self.rank = rank

    def forward(self, indices, offsets):
        emb_lookup = self.remote_emb_module.forward(indices, offsets)
        return self.fc(emb_lookup.to(self.rank))


def _run_trainer(remote_emb_module, rank):
    r"""
    Each trainer runs a forward pass which involves an embedding lookup on the
    parameter server and running nn.Linear locally. During the backward pass,
    DDP is responsible for aggregating the gradients for the dense part
    (nn.Linear) and distributed autograd ensures gradients updates are
    propagated to the parameter server.
    """
    # 每个训练进程的前向传播包括：在参数服务器上查找嵌入，并在本地运行 nn.Linear。
    # 反向传播时，DDP 聚合稠密部分的梯度，分布式自动求导将梯度更新传到参数服务器。

    # Setup the model.
    # 设置模型。
    model = HybridModel(remote_emb_module, rank)

    # Retrieve all model parameters as rrefs for DistributedOptimizer.
    # 以远程引用（RRef）的形式获取所有模型参数，供 DistributedOptimizer 使用。

    # Retrieve parameters for embedding table.
    # 获取嵌入表的参数。
    model_parameter_rrefs = model.remote_emb_module.remote_parameters()

    # model.fc.parameters() only includes local parameters.
    # model.fc.parameters() 只包含本地参数。
    # NOTE: Cannot call model.parameters() here,
    # 注意：这里不能调用 model.parameters()，
    # because this will call remote_emb_module.parameters(),
    # 因为它会调用 remote_emb_module.parameters()，
    # which supports remote_parameters() but not parameters().
    # 而远程模块支持 remote_parameters()，不支持 parameters()。
    for param in model.fc.parameters():
        model_parameter_rrefs.append(RRef(param))

    # Setup distributed optimizer
    # 设置分布式优化器。
    opt = DistributedOptimizer(
        optim.SGD,
        model_parameter_rrefs,
        lr=0.05,
    )

    criterion = torch.nn.CrossEntropyLoss()

    def get_next_batch(rank):
        for _ in range(10):
            num_indices = random.randint(20, 50)
            indices = torch.LongTensor(num_indices).random_(0, NUM_EMBEDDINGS)

            # Generate offsets.
            # 生成偏移量。
            offsets = []
            start = 0
            batch_size = 0
            while start < num_indices:
                offsets.append(start)
                start += random.randint(1, 10)
                batch_size += 1

            offsets_tensor = torch.LongTensor(offsets)
            target = torch.LongTensor(batch_size).random_(8).to(rank)
            yield indices, offsets_tensor, target

    # Train for 100 epochs
    # 训练 100 轮。
    for epoch in range(100):
        # create distributed autograd context
        # 创建分布式自动求导上下文。
        for indices, offsets, target in get_next_batch(rank):
            with dist_autograd.context() as context_id:
                output = model(indices, offsets)
                loss = criterion(output, target)

                # Run distributed backward pass
                # 执行分布式反向传播。
                dist_autograd.backward(context_id, [loss])

                # Tun distributed optimizer
                # 执行分布式优化器更新。
                opt.step(context_id)

                # Not necessary to zero grads as each iteration creates a different
                # 不需要清空梯度，因为每次迭代都会创建不同的
                # distributed autograd context which hosts different grads
                # 分布式自动求导上下文，各自保存不同的梯度。
        print("Training done for epoch {}".format(epoch))


def run_worker(rank, world_size):
    r"""
    A wrapper function that initializes RPC, calls the function, and shuts down
    RPC.
    """
    # 包装函数：初始化 RPC，调用目标函数，然后关闭 RPC。

    # We need to use different port numbers in TCP init_method for init_rpc and
    # init_rpc 和 init_process_group 的 TCP 初始化地址
    # init_process_group to avoid port conflicts.
    # 必须使用不同端口，避免端口冲突。
    rpc_backend_options = TensorPipeRpcBackendOptions()
    rpc_backend_options.init_method = "tcp://localhost:29501"

    # Rank 2 is master, 3 is ps and 0 and 1 are trainers.
    # rank 2 为主进程，rank 3 为参数服务器，rank 0 和 1 为训练进程。
    if rank == 2:
        rpc.init_rpc(
            "master",
            rank=rank,
            world_size=world_size,
            rpc_backend_options=rpc_backend_options,
        )

        remote_emb_module = RemoteModule(
            "ps",
            torch.nn.EmbeddingBag,
            args=(NUM_EMBEDDINGS, EMBEDDING_DIM),
            kwargs={"mode": "sum"},
        )

        # Run the training loop on trainers.
        # 在训练进程上运行训练循环。
        futs = []
        for trainer_rank in [0, 1]:
            trainer_name = "trainer{}".format(trainer_rank)
            fut = rpc.rpc_async(
                trainer_name, _run_trainer, args=(remote_emb_module, trainer_rank)
            )
            futs.append(fut)

        # Wait for all training to finish.
        # 等待所有训练完成。
        for fut in futs:
            fut.wait()
    elif rank <= 1:
        acc = torch.accelerator.current_accelerator()
        device = torch.device(acc)
        backend = torch.distributed.get_default_backend_for_device(device)
        torch.accelerator.device_index(rank)
        # Initialize process group for Distributed DataParallel on trainers.
        # 在训练进程上初始化 DDP 进程组。
        dist.init_process_group(
            backend=backend, rank=rank, world_size=2, init_method="tcp://localhost:29500"
        )

        # Initialize RPC.
        # 初始化 RPC。
        trainer_name = "trainer{}".format(rank)
        rpc.init_rpc(
            trainer_name,
            rank=rank,
            world_size=world_size,
            rpc_backend_options=rpc_backend_options,
        )

        # Trainer just waits for RPCs from master.
        # 训练进程只需等待主进程发来的 RPC。
    else:
        rpc.init_rpc(
            "ps",
            rank=rank,
            world_size=world_size,
            rpc_backend_options=rpc_backend_options,
        )
        # parameter server do nothing
        # 参数服务器无需执行额外操作。
        pass

    # block until all rpcs finish
    # 阻塞等待所有 RPC 完成。
    rpc.shutdown()
    
    # Clean up process group for trainers to avoid resource leaks
    # 清理训练进程的进程组，避免资源泄漏。
    if rank <= 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    # 2 trainers, 1 parameter server, 1 master.
    # 两个训练进程、一个参数服务器和一个主进程。
    world_size = 4
    _min_gpu_count = 2
    if not verify_min_gpu_count(min_gpus=_min_gpu_count):
        print(f"Unable to locate sufficient {_min_gpu_count} gpus to run this example. Exiting.")
        exit()
    mp.spawn(run_worker, args=(world_size,), nprocs=world_size, join=True)
    print("Distributed RPC example completed successfully.")
