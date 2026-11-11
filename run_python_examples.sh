#!/bin/bash
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
#   ./run_python_examples.sh
# * To run few specific examples:
# 运行几个指定示例：
#   ./run_python_examples.sh "dcgan,fast_neural_style"
#
# To test examples on CUDA accelerator, run as:
# 如需在 CUDA 加速设备上测试，按以下方式运行：
#   USE_CUDA=True ./run_python_examples.sh
#
# To test examples on hardware accelerator (CUDA, MPS, XPU, etc.), run as:
# 如需在 CUDA、MPS、XPU 等硬件加速设备上测试，按以下方式运行：
#   USE_ACCEL=True ./run_python_examples.sh
# NOTE: USE_ACCEL relies on torch.accelerator API and not all examples are converted
# 注意：USE_ACCEL 依赖 torch.accelerator API，目前并非所有示例
# to use it at the moment. Thus, expect failures using this flag on non-CUDA accelerators
# 都已完成迁移，因此在非 CUDA 加速设备上使用此标志时可能失败，
# and consider to run examples one by one.
# 可以考虑逐个运行示例。
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
#   VIRTUAL_ENV=".venv" ./run_python_examples.sh
# Script will remove environments it creates in a teardown step after execution of each example.
# 每个示例执行完毕后，脚本会在清理阶段移除自己创建的环境。

BASE_DIR="$(pwd)/$(dirname $0)"
source $BASE_DIR/utils.sh

# TODO: Leave only USE_ACCEL and drop USE_CUDA once all examples will be converted
# 待办：所有示例迁移到 torch.accelerator API 后，移除 USE_CUDA，仅保留 USE_ACCEL。
# to torch.accelerator API. For now, just add USE_ACCEL as an alias for USE_CUDA.
# 目前先将 USE_ACCEL 作为 USE_CUDA 的别名。
if [ -n "$USE_ACCEL" ]; then
  USE_CUDA=$USE_ACCEL
fi
USE_CUDA=${USE_CUDA:-False}
case $USE_CUDA in
  "True")
    echo "using cuda"
    CUDA=1
    CUDA_FLAG="--cuda"
    ACCEL_FLAG="--accel"
    ;;
  "False")
    echo "not using cuda"
    CUDA=0
    CUDA_FLAG=""
    ACCEL_FLAG=""
    ;;
  "")
    exit 1;
    ;;
esac

function dcgan() {
  uv run main.py --dataset fake $ACCEL_FLAG --dry-run || error "dcgan failed"
}

function exercises() {
  uv run 01_linear.py || error "linear exercise failed"
  uv run 02_tensor.py || error "tensor exercise failed"
}

function fast_neural_style() {
  if [ ! -d "saved_models" ]; then
    echo "downloading saved models for fast neural style"
    uv run download_saved_models.py
  fi
  test -d "saved_models" || { error "saved models not found"; return; }

  echo "running fast neural style model"
  uv run neural_style/neural_style.py eval --content-image images/content-images/amber.jpg --model saved_models/candy.pth --output-image images/output-images/amber-candy.jpg $ACCEL_FLAG || error "neural_style.py failed"
}

function imagenet() {
  if [[ ! -d "sample/val" || ! -d "sample/train" ]]; then
    mkdir -p sample/val/n
    mkdir -p sample/train/n
    curl -O "https://upload.wikimedia.org/wikipedia/commons/5/5a/Socks-clinton.jpg" || { error "couldn't download sample image for imagenet"; return; }
    mv Socks-clinton.jpg sample/train/n
    cp sample/train/n/* sample/val/n/
  fi
  uv run main.py --epochs 1 sample/ || error "imagenet example failed"
  uv run main.py --epochs 1 --gpu 0 sample/ || error "imagenet example failed"
}

function language_translation() {
  uv run -m spacy download en || error "couldn't download en package from spacy"
  uv run -m spacy download de || error "couldn't download de package from spacy"
  uv run main.py -e 1 --enc_layers 1 --dec_layers 1 --backend cpu --logging_dir output/ --dry_run || error "language translation example failed"
}

function mnist() {
  uv run main.py --epochs 1 --dry-run || error "mnist example failed"
}
function mnist_forward_forward() {
  uv run main.py --epochs 1 --no_accel || error "mnist forward forward failed"

}
function mnist_hogwild() {
  uv run main.py --epochs 1 --dry-run $CUDA_FLAG || error "mnist hogwild failed"
}

function mnist_rnn() {
  uv run main.py --epochs 1 --dry-run || error "mnist rnn example failed"
}

function regression() {
  uv run main.py --epochs 1 $CUDA_FLAG || error "regression failed"
}

function siamese_network() {
  uv run main.py --epochs 1 --dry-run || error "siamese network example failed"
}

function reinforcement_learning() {
  uv run reinforce.py || error "reinforcement learning reinforce failed"
  uv run actor_critic.py || error "reinforcement learning actor_critic failed"
}

function snli() {
  echo "installing 'en' model if not installed"
  uv run -m spacy download en || { error "couldn't download 'en' model needed for snli";  return; }
  echo "training..."
  uv run train.py --epochs 1 --dev_every 1 --no-bidirectional --dry-run || error "couldn't train snli"
}

function fx() {
  # uv run custom_tracer.py || error "fx custom tracer has failed" UnboundLocalError: local variable 'tabulate' referenced before assignment
  # 这条被注释的运行命令曾触发 UnboundLocalError：局部变量 tabulate 在赋值前被引用。
  uv run invert.py || error "fx invert has failed"
  uv run module_tracer.py || error "fx module tracer has failed"
  uv run primitive_library.py || error "fx primitive library has failed"
  uv run profiling_tracer.py || error "fx profiling tracer has failed"
  uv run replace_op.py || error "fx replace op has failed"
  uv run subgraph_rewriter_basic_use.py || error "fx subgraph has failed"
  uv run wrap_output_dynamically.py || error "vmap output dynamically has failed"
}

function super_resolution() {
  uv run main.py --upscale_factor 3 --batchSize 4 --testBatchSize 100 --nEpochs 1 --lr 0.001 $ACCEL_FLAG || error "super resolution failed"
  uv run super_resolve.py --input_image dataset/BSDS300/images/test/16077.jpg --model model_epoch_1.pth --output_filename out.png $ACCEL_FLAG || error "super resolution upscaling failed"
}

function time_sequence_prediction() {
  uv run generate_sine_wave.py || { error "generate sine wave failed";  return; }
  uv run train.py --steps 2 || error "time sequence prediction training failed"
}

function vae() {
  uv run main.py --epochs 1 || error "vae failed"
}

function vision_transformer() {
  uv run main.py --epochs 1 --dry-run || error "vision transformer example failed"
}

function word_language_model() {
  uv run main.py --epochs 1 --dry-run $ACCEL_FLAG || error "word_language_model failed"
  uv run generate.py $ACCEL_FLAG || error "word_language_model generate failed"
  for model in "RNN_TANH" "RNN_RELU" "LSTM" "GRU" "Transformer"; do
    uv run main.py --model $model --epochs 1 --dry-run $ACCEL_FLAG || error "word_language_model failed"
    uv run generate.py $ACCEL_FLAG || error "word_language_model generate failed"
  done
}

function gcn() {
  uv run main.py --epochs 1 --dry-run || error "graph convolutional network failed"
}

function gat() {
  uv run main.py --epochs 1 --dry-run || error "graph attention network failed"
}

eval "base_$(declare -f stop)"

function stop() {
  cd $BASE_DIR
  rm -rf dcgan/fake_samples_epoch_000.png \
    dcgan/netD_epoch_0.pth \
    dcgan/netG_epoch_0.pth \
    dcgan/real_samples.png \
    fast_neural_style/saved_models.zip \
    fast_neural_style/saved_models/ \
    imagenet/checkpoint.pth.tar \
    imagenet/lsun/ \
    imagenet/model_best.pth.tar \
    imagenet/sample/ \
    language_translation/output/ \
    snli/.data/ \
    snli/.vector_cache/ \
    snli/results/ \
    time_sequence_prediction/predict*.pdf \
    time_sequence_prediction/traindata.pt \
    word_language_model/model.pt \
    gcn/cora/ \
    gat/cora/ || error "couldn't clean up some files"

  git checkout fast_neural_style/images/output-images/amber-candy.jpg || error "couldn't clean up fast neural style image"

  base_stop "$1"
}

function run_all() {
  # cpp moved to `run_cpp_examples.sh```
  # C++ 示例已移至 run_cpp_examples.sh。
  run dcgan
  # distributed moved to `run_distributed_examples.sh`
  # 分布式示例已移至 run_distributed_examples.sh。
  run exercises
  run fast_neural_style
  run imagenet
  # language_translation
  # 语言翻译示例。
  run mnist
  run mnist_forward_forward
  run mnist_hogwild
  run mnist_rnn
  run regression
  run reinforcement_learning
  run siamese_network
  # run super_resolution - flaky
  # 运行超分辨率示例；该测试结果不稳定。
  run time_sequence_prediction
  run vae
  # vision_transformer - example broken see https://github.com/pytorch/examples/issues/1184 and https://github.com/pytorch/examples/pull/1258 for more details
  # vision_transformer 示例已损坏，详细情况见原文所列的问题和 PR。
  run word_language_model
  run fx
  run gcn
  run gat
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
  echo "Some python examples failed:"
  printf "$ERRORS\n"
  #Exit with error (0-255) in case of failure in one of the tests.
  # 任一测试失败时，以非零错误码退出，退出码范围为 0～255。
  exit 1

fi
