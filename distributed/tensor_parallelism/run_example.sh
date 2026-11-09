
# To run samples:
# 运行示例的方式：
# bash run_example.sh {file_to_run.py} {num_gpus}
# 运行方式：bash run_example.sh {要运行的 Python 文件} {GPU 数量}。
# where file_to_run = example to launch.  Default = 'fsdp_tp_example.py'
# file_to_run 为要启动的示例，默认为 fsdp_tp_example.py。
# num_gpus = num local gpus to use (must be at least 2). Default = 4
# num_gpus 是使用的本地 GPU 数量，至少为 2，默认值为 4。

# samples to run include:
# 可运行的示例包括：
# sequence_parallel_example.py
# tensor_parallel_example.py
# fsdp_tp_example.py

echo "Launching ${1:-fsdp_tp_example.py} with ${2:-4} gpus"
torchrun --nnodes=1 --nproc_per_node=${2:-4} --rdzv_id=101 --rdzv_endpoint="localhost:5972" ${1:-fsdp_tp_example.py}
