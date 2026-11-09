#!/bin/bash
# Usage: bash run_example.sh {file_to_run.py} {num_gpus}
# 用法：bash run_example.sh {要运行的 Python 文件} {GPU 数量}。
# where file_to_run = example to run. Default = 'main.py'
# file_to_run 为要运行的示例，默认为 main.py。
# num_gpus = num local gpus to use. Default = 16
# num_gpus 是使用的本地 GPU 数量，默认值为 16。

# samples to run include:
# 可运行的示例包括：
# main.py

echo "Launching ${1:-main.py} with ${2:-16} gpus"
torchrun --standalone --nproc_per_node=${2:-16} ${1:-main.py}
