import os

import torch
import torch.distributed.autograd as dist_autograd
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp
import torch.optim as optim
from torch.distributed.optim import DistributedOptimizer

import rnn


def _run_trainer():
    r"""
    The trainer creates a distributed RNNModel and a DistributedOptimizer. Then,
    it performs training using random input data.
    """
    # 训练进程创建分布式 RNNModel 和 DistributedOptimizer，
    # 再使用随机输入数据训练。
    batch = 5
    ntoken = 7
    ninp = 2

    nhid = 3
    nindices = 6
    nlayers = 4
    hidden = (
        torch.randn(nlayers, nindices, nhid),
        torch.randn(nlayers, nindices, nhid)
    )

    model = rnn.RNNModel('ps', ntoken, ninp, nhid, nlayers)

    # setup distributed optimizer
    # 设置分布式优化器。
    opt = DistributedOptimizer(
        optim.SGD,
        model.parameter_rrefs(),
        lr=0.05,
    )

    criterion = torch.nn.CrossEntropyLoss()

    def get_next_batch():
        for _ in range(5):
            data = torch.LongTensor(batch, nindices) % ntoken
            target = torch.LongTensor(batch, ntoken) % nindices
            yield data, target

    # train for 10 iterations
    # 训练 10 次迭代。
    for epoch in range(10):
        # create distributed autograd context
        # 创建分布式自动求导上下文。
        for data, target in get_next_batch():
            with dist_autograd.context() as context_id:
                hidden[0].detach_()
                hidden[1].detach_()
                output, hidden = model(data, hidden)
                loss = criterion(output, target)
                # run distributed backward pass
                # 执行分布式反向传播。
                dist_autograd.backward(context_id, [loss])
                # run distributed optimizer
                # 执行分布式优化器更新。
                opt.step(context_id)
                # not necessary to zero grads as each iteration creates a different
                # 无需清空梯度，因为每次迭代都会创建不同的
                # distributed autograd context which hosts different grads
                # 分布式自动求导上下文，各自保存不同的梯度。
        print("Training epoch {}".format(epoch))


def run_worker(rank, world_size):
    r"""
    A wrapper function that initializes RPC, calls the function, and shuts down
    RPC.
    """
    # 包装函数：初始化 RPC，调用目标函数，然后关闭 RPC。
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    if rank == 1:
        rpc.init_rpc("trainer", rank=rank, world_size=world_size)
        _run_trainer()
    else:
        rpc.init_rpc("ps", rank=rank, world_size=world_size)
        # parameter server does nothing
        # 参数服务器无需额外操作。
        pass

    # block until all rpcs finish
    # 阻塞等待所有 RPC 完成。
    rpc.shutdown()


if __name__ == "__main__":
    world_size = 2
    mp.spawn(run_worker, args=(world_size, ), nprocs=world_size, join=True)
