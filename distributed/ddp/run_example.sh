# /bin/bash
# bash run_example.sh {file_to_run.py} {num_gpus}
# 运行方式：bash run_example.sh {要运行的 Python 文件} {GPU 数量}。
# where file_to_run = example to run. Default = 'example.py'
# file_to_run 表示要运行的示例，默认为 example.py。
# num_gpus = num local gpus to use (must be at least 2). Default = 2
# num_gpus 表示使用的本地 GPU 数量，至少为 2，默认值为 2。

# samples to run include:
# 可运行的示例包括：
# example.py

echo "Launching ${1:-example.py} with ${2:-2} gpus"
torchrun --nnodes=1 --nproc_per_node=${2:-2} --rdzv_id=101 --rdzv_endpoint="localhost:5972" ${1:-example.py}
