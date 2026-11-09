from pathlib import Path
from datetime import datetime
import torch
import time

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    StateDictType,
    FullStateDictConfig,  # general model non-sharded, non-flattened params
    # 通用模型参数：未分片、未展平。
    LocalStateDictConfig,  # flattened params, usable only by FSDP
    # 展平后的参数，仅供 FSDP 使用。
    # ShardedStateDictConfig, # un-flattened param but shards, usable by other parallel schemes.
    # ShardedStateDictConfig 对应未展平但已分片的参数，也可用于其他并行方案。
)

from torch.distributed.checkpoint import (
    FileSystemReader,
    FileSystemWriter,
    save_state_dict,
    load_state_dict,
)
from torch.distributed.checkpoint.default_planner import (
    DefaultSavePlanner,
    DefaultLoadPlanner,
)


from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType
import torch.distributed.checkpoint as dist_cp
import torch.distributed as dist


def get_date_of_run():
    """create date and time for file save uniqueness
    example: 2022-05-07-08:31:12_PM'
    """
    # 生成日期时间，保证保存文件名唯一。
    # 例如：2022-05-07-08:31:12_PM。
    date_of_run = datetime.now().strftime("%Y-%m-%d-%I:%M:%S_%p")
    print(f"--> current date and time of run = {date_of_run}")
    return date_of_run


# create singleton saving policies to avoid making over and over
# 创建可复用的保存策略实例，避免重复创建。
fullstate_save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)


def load_model_sharded(model, rank, cfg, verbose=True):
    # torch.manual_seed(103)
    folder_name = (
        cfg.dist_checkpoint_root_folder
        + "/"
        + cfg.dist_checkpoint_folder
        + "-"
        + cfg.model_name
    )

    load_dir = Path.cwd() / folder_name

    if not load_dir.exists():
        if rank == 0:
            print(f"No sharded_state_dict checkpoint directory found...skipping")
        return

    reader = FileSystemReader(load_dir)

    with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
        checkpoint = model.state_dict()
        if rank == 0:
            ck = checkpoint.keys()
            print(f" checkpoint key len = {len(ck)} and \n keys =  {ck}")

        dist_cp.load_state_dict(
            state_dict=checkpoint,
            storage_reader=reader,
        )
        if rank == 0:
            print(f"checkpoint after load_state_dict()")
            ck = checkpoint.keys()
            print(f" checkpoint key len = {len(ck)} and \n keys =  {ck}")
        model.load_state_dict(checkpoint)
    if rank == 0:
        print(f"Sharded state checkpoint loaded from {load_dir}")


def save_model_and_optimizer_sharded(model, rank, cfg,optim=None, verbose=True):
    """save model and optimizer via sharded_state_dict to save_dir"""
    # 通过 sharded_state_dict 将模型和优化器保存到 save_dir。
    folder_name = (
        cfg.dist_checkpoint_root_folder
        + "/"
        + cfg.dist_checkpoint_folder
        + "-"
        + cfg.model_name
    )

    save_dir = Path.cwd() / folder_name
    if rank == 0:
        print(f"Saving model to {save_dir}")

    distributed_writer = dist_cp.FileSystemWriter(
        save_dir,
    )
    t0 = time.perf_counter()

    with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):

        state_dict = {"model": model.state_dict()}
        if optim is not None:
            state_dict["optim"] = FSDP.optim_state_dict(model, optim)

        dist_cp.save_state_dict(
            state_dict=state_dict,
            storage_writer=distributed_writer,
            planner=DefaultSavePlanner(),

        )
    dist.barrier()
    t1 = time.perf_counter()
    if rank == 0:
        print(f"Sharded state checkpoint saved to {save_dir}")
        print(
            f"Checkpoint Time = {t1-t0:.4f}\n using {cfg.save_using_num_threads=} total threads"
        )

def save_model_checkpoint(
    model,
    optimizer,
    rank,
    cfg,
    epoch=1,
):
    """saving model via rank0 cpu streaming and full_state_dict"""
    # 通过 rank 0 的 CPU 流式处理和 full_state_dict 保存模型。

    # saving with rank0 cpu
    # 通过 rank 0 进程的 CPU 保存。
    if not cfg.checkpoint_type == StateDictType.FULL_STATE_DICT:
        print(f" unable to handle checkpoint type {cfg.checkpoint_type}, aborting")

    with FSDP.state_dict_type(
        model, StateDictType.FULL_STATE_DICT, fullstate_save_policy
    ):
        cpu_state = model.state_dict()

    if cfg.verbose:
        print(f"saving process: rank {rank}  done w model state_dict\n")


    if rank == 0:
        print(f"--> saving model ...")
        # create save path
        # 创建保存路径。
        save_dir = Path.cwd() / cfg.checkpoint_folder
        save_dir.mkdir(parents=True, exist_ok=True)
        save_name = cfg.model_save_name + "-" + str(epoch) + ".pt"
        save_full_path = str(save_dir) + "/" + save_name

        # save model
        # 保存模型。
        torch.save(cpu_state, save_full_path)

        if cfg.verbose:
            print(f"model checkpoint saved for epoch {epoch} at {save_full_path}\n")



def load_model_checkpoint(model, rank, cfg, verbose=True):
    """load local checkpoint to rank0 cpu
    must be called * before * passing to FSDP"""
    # 将本地检查点加载到 rank 0 的 CPU。
    # 必须在将模型交给 FSDP 包装之前调用。

    if rank != 0:
        return

    # where is the checkpoint at...
    # 检查点所在位置。
    full_state_dict_model_path = (
        Path.cwd() / cfg.checkpoint_folder / cfg.checkpoint_model_filename
    )
    # is it present...
    # 检查文件是否存在。
    if not full_state_dict_model_path.is_file():
        print(
            f"model checkpoint {full_state_dict_model_path} not present. Returning..."
        )
        return


    model_checkpoint = torch.load(full_state_dict_model_path)
    # integrate into loaded model
    # 整合到已加载的模型中。
    model.load_state_dict(model_checkpoint)

    if cfg.verbose:
        print(f"model checkpoint loaded to rank0 cpu")


def save_optimizer_checkpoint(model, optimizer, rank, cfg, epoch=1):
    """save optimizer state via full state dict"""
    # 通过完整状态字典保存优化器状态。

    if cfg.verbose:
        print(f"--> optim state call on rank {rank}\n")

    # pull all sharded optimizer states to rank0 cpu...
    # 将所有分片的优化器状态汇集到 rank 0 的 CPU。

    optim_state = FSDP.full_optim_state_dict(model, optimizer)

    if cfg.verbose:
        print(f"optim state dict ready on {rank} and len of {len(optim_state)}\n")

    if rank == 0:
        save_dir = Path.cwd() / cfg.checkpoint_folder
        save_dir.mkdir(parents=True, exist_ok=True)

        opt_save_name = (
            cfg.optimizer_name + "-" + cfg.model_save_name + "-" + str(epoch) + ".pt"
        )
        opt_save_full_path = save_dir / opt_save_name

        print(f"--> saving optimizer state...")

        torch.save(optim_state, opt_save_full_path)

        print(f"--> saved {opt_save_full_path} to disk")


def load_optimizer_checkpoint(model, optimizer, rank, cfg):
    """load an fdsp optimizer full_state checkpoint using scatter method
    this ensures only rank 0 loads the optimizer state dict and scatters to other ranks
    """
    # 使用 scatter 方法加载 FSDP 优化器的完整状态检查点。
    # 仅 rank 0 加载优化器状态字典，再分发给其他进程。

    opt_file_path = Path.cwd() / cfg.checkpoint_folder / cfg.optimizer_checkpoint_file

    if not opt_file_path.is_file():
        print(
            f"warning - optimizer checkpoint not present {opt_file_path}. Returning. "
        )
        return

    full_osd = None

    if rank == 0:
        full_osd = torch.load(opt_file_path)

        if cfg.verbose:
            print(f"loaded full osd on rank 0")

    # called from all ranks, though only rank0 has a valid param for full_osd
    # 所有进程都调用此函数，但只有 rank 0 的 full_osd 参数有效。
    sharded_osd = FSDP.scatter_full_optim_state_dict(full_osd, model)

    if cfg.verbose:
        print(f"optimizer shard loaded on rank {rank}")



def load_distributed_model_checkpoint(model, rank, cfg):
    if cfg.checkpoint_type == StateDictType.LOCAL_STATE_DICT:
        print(f"loading distributed checkpoint, rank {rank}...")
        folder_name = (
            cfg.dist_checkpoint_root_folder
            + "/"
            + cfg.dist_checkpoint_folder
            + "-"
            + cfg.model_name
        )

        checkdir = Path.cwd() / folder_name

        if not checkdir.exists():
            if rank == 0:
                print(f"No checkpoint directory found...skipping")
            return


        reader = FileSystemReader(checkdir)

        with FSDP.state_dict_type(
            model,
            StateDictType.LOCAL_STATE_DICT,
        ):
            state_dict = model.state_dict()
            load_state_dict(state_dict, reader)
            model.load_state_dict(state_dict)

        print(f"--> local state loaded on rank {rank}")

        return


def save_distributed_model_checkpoint(model, rank, cfg, epoch=1):
    # distributed checkpoint saving
    # 保存分布式检查点。

    # confirm type of checkpoint and save
    # 确认检查点类型并保存。
    if cfg.checkpoint_type == StateDictType.LOCAL_STATE_DICT:
        # create writer to current path
        # 为当前路径创建写入器。
        folder_name = (
            cfg.dist_checkpoint_root_folder
            + "/"
            + cfg.dist_checkpoint_folder
            + "-"
            + cfg.model_name
        )
        save_dir = Path.cwd() / folder_name

        writer = FileSystemWriter(
            save_dir,
        )

        with FSDP.state_dict_type(
            model,
            StateDictType.LOCAL_STATE_DICT,
        ):
            state_dict = model.state_dict()


        # write out distributed checkpoint
        # 写入分布式检查点。
        save_state_dict(state_dict, writer)

        return
