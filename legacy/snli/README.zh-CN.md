# 基于 PyTorch 和 SNLI 的自然语言推理训练

[English original](README.md)

> 中文说明；代码块中的命令、代码及程序输出保留原样。

## 概述

本目录提供 Python 脚本，使用斯坦福自然语言推理（SNLI）语料库训练 `SNLIClassifier` 自然语言推理（NLI）模型。模型判断两个陈述之间的关系：蕴含、矛盾或中立。

## 依赖

安装所需的 Python 库：

```bash
pip install -r requirements.txt
```

`requirements.txt` 包含：

```
torch
torchtext
spacy
```

## 使用方法

启动训练：

```bash
python train.py --lower --word-vectors [PATH_TO_WORD_VECTORS] --vector-cache [PATH_TO_VECTOR_CACHE] --epochs [NUMBER_OF_EPOCHS] --batch-size [BATCH_SIZE] --save-path [PATH_TO_SAVE_MODEL] --gpu [GPU_NUMBER]
```

## 训练

脚本按小批次读取数据，训练指定轮数，并将验证集表现最好的模型以 `.pt` 文件保存到指定目录。

## 脚本说明

- `model.py`：定义 `SNLIClassifier` 模型及辅助类。
- `util.py`：提供创建目录和解析命令行参数等工具函数。

## 注意

请确保工作目录中包含 `model.py` 和 `util.py`。
