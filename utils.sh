#!/usr/bin/env bash
# This script contains utility functions and initialize exmaple scripts.
# 此脚本包含工具函数，并初始化各示例运行脚本，
# Eg: run_python_examples.sh, run_distributed_examples.sh
# 例如 run_python_examples.sh 和 run_distributed_examples.sh。

BASE_DIR="$(pwd)/$(dirname $0)"
EXAMPLES=$(echo $1 | sed -e 's/ //g')

# Redirect 'python' calls to 'python3'
# 将 python 调用重定向到 python3。
python() {
    command python3 "$@"
}

ERRORS=${ERRORS-""}

function error() {
  ERR=$1
  if [ "" == "$ERRORS" ]; then
    ERRORS="$ERR"
  else
    ERRORS="$ERRORS\n$ERR"
  fi
}

function start() {
  EXAMPLE=$1
  cd $BASE_DIR/$EXAMPLE || { error "$EXAMPLE: no such example"; return 1; }
  echo "Install dependencies for $EXAMPLE"
  # Setting VIRTUAL_ENV=.venv externally will create uv virtual environment
  # 在外部设置 VIRTUAL_ENV=.venv 后，会在 start() 中为每个示例
  # for each sample in start() and remove it in stop(). Note that this environment
  # 创建 uv 虚拟环境，并在 stop() 中移除。注意，此环境
  # variable also forces other uv commands such as `uv pip...` and `uv run...` to
  # 变量也会强制 uv pip、uv run 等其他 uv 命令
  # use the specified environment.
  # 使用指定环境。
  if [ "$VIRTUAL_ENV" = ".venv" ]; then
    uv venv || { error "$EXAMPLE: failed to create virtual environment"; return 1; }
  fi
  uv pip install -r requirements.txt $PIP_INSTALL_ARGS || { error "$EXAMPLE: failed to install requirements"; return 1; }
  echo "Running example: $EXAMPLE"
}

function stop() {
  EXAMPLE=$1
  if [ "$VIRTUAL_ENV" = ".venv" ]; then
    cd $BASE_DIR/$EXAMPLE && rm -rf .venv
  fi
}

function run() {
  EXAMPLE=$1
  if start $EXAMPLE; then
    # drop trailing slash (occurs due to auto completion in bash interactive mode)
    # 去掉路径末尾的斜杠，该斜杠可能由 Bash 交互模式的自动补全产生。
    # replace slashes with underscores: this allows to call nested examples
    # 将斜杠替换为下划线，以便调用嵌套目录中的示例。
    EXAMPLE_FN=$(echo $EXAMPLE | sed "s@/\$@@" | sed 's@/@_@g')
    $EXAMPLE_FN
  fi
  stop $EXAMPLE
}
