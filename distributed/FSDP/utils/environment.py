# Copyright (c) 2022 Meta Platforms, Inc. and its affiliates.
# 版权所有 © 2022 Meta Platforms, Inc. 及其关联方。
# All rights reserved.
# 保留所有权利。
#

# This is a simple check to confirm that your current server has full bfloat support -
# 这里简单检查当前服务器是否完整支持 bfloat16，
#  both GPU native support, and Network communication support.
# 包括 GPU 原生支持和网络通信支持。

# Be warned that if you run on V100 without a check like this, you will be running without native Bfloat16
# 如果不做检查就在 V100 上运行，将缺少原生 bfloat16
# support and will find significant performance degradation (but it will not complain via an error).
# 支持，性能会明显下降，但不会因此报错。
# Hence the reason for a checker!
# 这就是需要进行检查的原因。

from pkg_resources import packaging
import torch
import torch.cuda.nccl as nccl
import torch.distributed as dist

# global flag that confirms ampere architecture, cuda version and
# 全局标志：检查 Ampere 架构、CUDA 版本和
# nccl version to verify bfloat16 native support is ready
# NCCL 版本，以确认已具备 bfloat16 原生支持。

def bfloat_support():
    return (
        torch.version.cuda
        and torch.cuda.is_bf16_supported()
        and packaging.version.parse(torch.version.cuda).release >= (11, 0)
        and dist.is_nccl_available()
        and nccl.version() >= (2, 10)
    )
