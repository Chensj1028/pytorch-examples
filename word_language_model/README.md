# Word-level Language Modeling using RNN and Transformer

使用 RNN 和 Transformer 进行词级语言建模

This example trains a multi-layer RNN (Elman, GRU, or LSTM) or Transformer on a language modeling task. By default, the training script uses the Wikitext-2 dataset, provided.
The trained model can then be used by the generate script to generate new text.

本示例训练多层 RNN（Elman、GRU 或 LSTM）或 Transformer 来完成语言建模任务。训练脚本默认使用随示例提供的 Wikitext-2 数据集。
训练后的模型可以通过生成脚本产生新文本。

```bash
python main.py --accel --epochs 6           # Train a LSTM on Wikitext-2.
# 在 WikiText-2 上训练 LSTM。
python main.py --accel --epochs 6 --tied    # Train a tied LSTM on Wikitext-2.
# 在 WikiText-2 上训练共享权重的 LSTM。
python main.py --accel --tied               # Train a tied LSTM on Wikitext-2for 40 epochs.
# 在 WikiText-2 上训练共享权重的 LSTM，共 40 轮。
python main.py --accel --epochs 6 --model Transformer --lr 5
                                            # Train a Transformer model on Wikitext-2.
                                            # 在 WikiText-2 上训练 Transformer 模型。
python main.py --accel --epochs 6 --model Transformer --use-optimizer --lr 0.001
                                            # Train a Transformer model with AdamW optimizer on Wikitext-2.
                                            # 在 WikiText-2 上使用 AdamW 优化器训练 Transformer 模型。

python generate.py --accel                  # Generate samples from the default model checkpoint.
# 从默认模型检查点生成样本。
```

> [!NOTE] 
> Example supports running on acceleration devices (CUDA, MPS, XPU)

> [!NOTE]
> 本示例支持使用 CUDA、MPS、XPU 加速设备运行。

The model uses the `nn.RNN` module (and its sister modules `nn.GRU` and `nn.LSTM`) or Transformer module (`nn.TransformerEncoder` and `nn.TransformerEncoderLayer`) which will automatically use the cuDNN backend if run on CUDA with cuDNN installed.

模型使用 `nn.RNN` 模块及其相关模块 `nn.GRU`、`nn.LSTM`，或者 Transformer 模块 `nn.TransformerEncoder`、`nn.TransformerEncoderLayer`。RNN 系列模块在 CUDA 环境中满足条件且已安装 cuDNN 时，会自动使用 cuDNN 后端。

During training, if a keyboard interrupt (Ctrl-C) is received, training is stopped and the current model is evaluated against the test dataset.

训练过程中按 Ctrl-C 触发键盘中断时，程序会停止训练，并用测试集评估当前模型。

The `main.py` script accepts the following arguments:

`main.py` 脚本接受以下参数：

```bash
optional arguments:
  -h, --help            show this help message and exit
  --data DATA           location of the data corpus
  --model MODEL         type of network (RNN_TANH, RNN_RELU, LSTM, GRU, Transformer)
  --emsize EMSIZE       size of word embeddings
  --nhid NHID           number of hidden units per layer
  --nlayers NLAYERS     number of layers
  --lr LR               initial learning rate
  --clip CLIP           gradient clipping
  --epochs EPOCHS       upper epoch limit
  --batch_size N        batch size
  --bptt BPTT           sequence length
  --dropout DROPOUT     dropout applied to layers (0 = no dropout)
  --tied                tie the word embedding and softmax weights
  --seed SEED           random seed
  --accel               use accelerator
  --log-interval N      report interval
  --save SAVE           path to save the final model
  --onnx-export ONNX_EXPORT
                        path to export the final model in onnx format
  --nhead NHEAD         the number of heads in the encoder/decoder of the transformer model
  --dry-run             verify the code and the model
  --use-optimizer       specify whether to use an AdamW optimizer
```

With these arguments, a variety of models can be tested.
As an example, the following arguments produce slower but better models:

可以通过这些参数测试不同模型。例如，下列配置训练较慢，但通常能得到更好的模型：

```bash
python main.py --accel --emsize 650 --nhid 650 --dropout 0.5 --epochs 40
python main.py --accel --emsize 650 --nhid 650 --dropout 0.5 --epochs 40 --tied
python main.py --accel --emsize 1500 --nhid 1500 --dropout 0.65 --epochs 40
python main.py --accel --emsize 1500 --nhid 1500 --dropout 0.65 --epochs 40 --tied
```
