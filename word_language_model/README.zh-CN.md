# 使用 RNN 和 Transformer 进行词级语言建模

[English original](README.md)

> 中文说明；代码块中的命令、代码及程序输出保留原样。

本示例训练多层 RNN（Elman、GRU 或 LSTM）或 Transformer 来完成语言建模任务。训练脚本默认使用随示例提供的 Wikitext-2 数据集。
训练后的模型可以通过生成脚本产生新文本。

```bash
python main.py --accel --epochs 6           # Train a LSTM on Wikitext-2.
python main.py --accel --epochs 6 --tied    # Train a tied LSTM on Wikitext-2.
python main.py --accel --tied               # Train a tied LSTM on Wikitext-2for 40 epochs.
python main.py --accel --epochs 6 --model Transformer --lr 5
                                            # Train a Transformer model on Wikitext-2.
python main.py --accel --epochs 6 --model Transformer --use-optimizer --lr 0.001
                                            # Train a Transformer model with AdamW optimizer on Wikitext-2.

python generate.py --accel                  # Generate samples from the default model checkpoint.
```

> [!NOTE]
> 本示例支持使用 CUDA、MPS、XPU 加速设备运行。

模型使用 `nn.RNN` 模块及其相关模块 `nn.GRU`、`nn.LSTM`，或者 Transformer 模块 `nn.TransformerEncoder`、`nn.TransformerEncoderLayer`。RNN 系列模块在 CUDA 环境中满足条件且已安装 cuDNN 时，会自动使用 cuDNN 后端。

训练过程中按 Ctrl-C 触发键盘中断时，程序会停止训练，并用测试集评估当前模型。

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

可以通过这些参数测试不同模型。例如，下列配置训练较慢，但通常能得到更好的模型：

```bash
python main.py --accel --emsize 650 --nhid 650 --dropout 0.5 --epochs 40
python main.py --accel --emsize 650 --nhid 650 --dropout 0.5 --epochs 40 --tied
python main.py --accel --emsize 1500 --nhid 1500 --dropout 0.65 --epochs 40
python main.py --accel --emsize 1500 --nhid 1500 --dropout 0.65 --epochs 40 --tied
```
