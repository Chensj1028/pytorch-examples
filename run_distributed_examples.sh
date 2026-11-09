#!/usr/bin/env bash
#
# This script runs through the code in each of the python examples.
# 此脚本依次执行各个 Python 示例中的代码。
# The purpose is just as an integration test, not to actually train models in any meaningful way.
# 目的仅是进行集成测试，而非实际训练出有意义的模型。
# For that reason, most of these set epochs = 1 and --dry-run.
# 因此，大多数示例将 epochs 设为 1，并启用 --dry-run。
#
# Optionally specify a comma separated list of examples to run. Can be run as:
# 可选：以逗号分隔的列表指定示例，运行方式如下：
# * To run all examples:
# 运行全部示例：
#   ./run_distributed_examples.sh
# * To run specific example:
# 运行指定示例：
#   ./run_distributed_examples.sh "distributed/tensor_parallelism,distributed/ddp"
#
# To test examples on CUDA accelerator, run as:
# 如需在 CUDA 加速设备上测试，按以下方式运行：
#   USE_CUDA=True ./run_distributed_examples.sh
#
# Script requires uv to be installed. When executed, script will install prerequisites from
# 运行前需要安装 uv。执行时，脚本根据每个示例的
# `requirements.txt` for each example. If ran within activated virtual environment (uv venv,
# requirements.txt 安装依赖。如果在已激活的虚拟环境（uv venv、
# python -m venv, conda) this might reinstall some of the packages. To change pip installation
# python -m venv、conda）中运行，部分包可能会被重新安装。若要修改 pip 安装源
# index or to pass additional pip install options, run as:
# 或传入额外安装选项，按以下方式运行：
#   PIP_INSTALL_ARGS="--pre -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html" \
#     ./run_python_examples.sh
#
# To force script to create virtual environment for each example, run as:
# 若要强制为每个示例创建独立虚拟环境，按以下方式运行：
#   VIRTUAL_ENV=".venv" ./run_distributed_examples.sh
# Script will remove environments it creates in a teardown step after execution of each example.
# 每个示例执行完毕后，脚本会在清理阶段移除自己创建的环境。

BASE_DIR="$(pwd)/$(dirname $0)"
source $BASE_DIR/utils.sh

USE_CUDA=${USE_CUDA:-False}
case $USE_CUDA in
  "True")
    echo "using cuda"
    CUDA=1
    CUDA_FLAG="--cuda"
    ;;
  "False")
    echo "not using cuda"
    CUDA=0
    CUDA_FLAG=""
    ;;
  "")
    exit 1;
    ;;
esac

function distributed_tensor_parallelism() {
    uv run bash run_example.sh tensor_parallel_example.py || error "tensor parallel example failed"
    uv run bash run_example.sh sequence_parallel_example.py || error "sequence parallel example failed"
    uv run bash run_example.sh fsdp_tp_example.py || error "2D parallel example failed"
}

function distributed_FSDP2() {
    uv run bash run_example.sh example.py || error "FSDP2 example failed"
}

function distributed_ddp() {
    uv run bash run_example.sh example.py || error "ddp example failed"
}

function distributed_minGPT-ddp() {
  uv run bash run_example.sh mingpt/main.py || error "minGPT example failed"
}

function distributed_rpc_ddp_rpc() {
    uv run main.py || error "ddp_rpc example failed"
}

function distributed_rpc_rnn() {
    uv run main.py || error "rpc_rnn example failed"
}

function run_all() {
  run distributed/tensor_parallelism
  run distributed/ddp
  run distributed/minGPT-ddp
  run distributed/rpc/ddp_rpc
  run distributed/rpc/rnn
}

# by default, run all examples
# 默认运行所有示例。
if [ "" == "$EXAMPLES" ]; then
  run_all
else
  for i in $(echo $EXAMPLES | sed "s/,/ /g")
  do
    echo "Starting $i"
    run $i
    echo "Finished $i, status $?"
  done
fi

if [ "" == "$ERRORS" ]; then
  echo "Completed successfully with status $?"
else
  echo "Some distributed examples failed:"
  printf "$ERRORS\n"
  #Exit with error (0-255) in case of failure in one of the tests.
  # 任一测试失败时，以非零错误码退出，退出码范围为 0～255。
  exit 1

fi
