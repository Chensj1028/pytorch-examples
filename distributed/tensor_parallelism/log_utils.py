import logging
import torch

logging.basicConfig(
    format="%(asctime)s %(message)s", datefmt="%m/%d/%Y %I:%M:%S %p", level=logging.INFO
)

def get_logger():
    return logging.getLogger(__name__)


def rank_log(_rank, logger, msg):
    """helper function to log only on global rank 0"""
    # 仅在全局 rank 为 0 时输出日志的辅助函数。
    if _rank == 0:
        logger.info(f" {msg}")


def verify_min_gpu_count(min_gpus: int = 2) -> bool:
    """ verification that we have at least 2 gpus to run dist examples """
    # 检查至少有两个 GPU，以运行分布式示例。
    has_gpu = torch.accelerator.is_available()
    gpu_count = torch.accelerator.device_count()
    return has_gpu and gpu_count >= min_gpus
