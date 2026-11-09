# Language Translation

语言翻译

This example shows how one might use transformers for language translation. In particular, this implementation is loosely based on the [Attention is All You Need paper](https://arxiv.org/abs/1706.03762).

本示例展示如何使用 Transformer 进行语言翻译，实现思路主要参考论文 [Attention is All You Need](https://arxiv.org/abs/1706.03762)。

## Requirements

环境依赖

We will need a tokenizer for our languages. Torchtext does include a tokenizer for English, but unfortunately, we will need more languages then that. We can get these tokenizers via ```spacy```

首先需要为所用语言准备分词器。Torchtext 提供英语分词器，但本例还需要其他语言的分词器，可以通过 `spacy` 获取：

```bash
python3 -m spacy download <language>
python3 -m spacy download en
python3 -m spacy download de
```

Spacy supports many languages. For a full accounting of supported languages, please look [here](https://spacy.io/usage/models). This example will default from German to English.

spaCy 支持多种语言，完整列表见[语言模型文档](https://spacy.io/usage/models)。本示例默认将德语翻译成英语。

Torchtext is also required:

还需要安装 Torchtext：

```bash
pip install torchtext
```

Just running these commands will get you started:

执行以下命令即可开始准备环境：

```bash
pip install -r requirements.txt
python3 -m spacy download <language-you-want>
```

## Usage

使用方法

This example contains a lot of flags that you can set to change the behavior / training of the module. You can see all of them by running:

本示例提供多种参数来调整模型行为及训练方式，可以运行以下命令查看：

```bash
python3 main.py -h
```

But in general, all of the settings have "sensible" defaults; however, the default translation is to translate from German to English. To *train* the model, you only need to run the following command, but there is also an example for how to use any language you want:

各项配置均提供默认值，默认翻译方向为德语到英语。训练模型只需执行下面的命令；代码块中也给出了指定其他语言的示例：

```bash
python3 main.py
python3 main.py --src en --tgt fr # For english to french translation
# 用于英语到法语的翻译。
```

For model inference, you can use this command:

使用以下命令进行模型推理：

```bash
python3 main.py --inference --model_path <path-to-model>
```

After some loading time, this will open an interactive interface where you can type in whatever sentence you are interested in translating.

加载完成后，程序会打开交互界面，你可以输入希望翻译的句子。
