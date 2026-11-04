# 语言翻译

[English original](README.md)

> 中文说明；代码块中的命令、代码及程序输出保留原样。

本示例展示如何使用 Transformer 进行语言翻译，实现思路主要参考论文 [Attention is All You Need](https://arxiv.org/abs/1706.03762)。

## 环境依赖

首先需要为所用语言准备分词器。Torchtext 提供英语分词器，但本例还需要其他语言的分词器，可以通过 `spacy` 获取：

```bash
python3 -m spacy download <language>
python3 -m spacy download en
python3 -m spacy download de
```

spaCy 支持多种语言，完整列表见[语言模型文档](https://spacy.io/usage/models)。本示例默认将德语翻译成英语。

还需要安装 Torchtext：

```bash
pip install torchtext
```

执行以下命令即可开始准备环境：

```bash
pip install -r requirements.txt
python3 -m spacy download <language-you-want>
```

## 使用方法

本示例提供多种参数来调整模型行为及训练方式，可以运行以下命令查看：

```bash
python3 main.py -h
```

各项配置均提供默认值，默认翻译方向为德语到英语。训练模型只需执行下面的命令；代码块中也给出了指定其他语言的示例：

```bash
python3 main.py
python3 main.py --src en --tgt fr # For english to french translation
```

使用以下命令进行模型推理：

```bash
python3 main.py --inference --model_path <path-to-model>
```

加载完成后，程序会打开交互界面，你可以输入希望翻译的句子。
