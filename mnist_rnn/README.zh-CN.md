# 使用 RNN 识别 MNIST 手写数字

[English original](README.md)

> 中文说明；代码块中的命令、代码及程序输出保留原样。

## 示例目的

创建一个 PyTorch 示例，对应 TensorFlow Keras 官方使用 MNIST 的 [RNN 示例](https://www.tensorflow.org/guide/keras/rnn)。

```bash
pip install -r requirements.txt
python main.py
# CUDA_VISIBLE_DEVICES=2 python main.py  # to specify GPU id to ex. 2
```

```bash
optional arguments:
  -h, --help            show this help message and exit
  --batch_size          input batch_size for training (default:64)
  --testing_batch_size  input batch size for testing (default: 1000)
  --epochs EPOCHS       number of epochs to train (default: 14)
  --lr LR               learning rate (default: 0.1)
  --gamma               learning rate step gamma (default: 0.7)
  --accel               enables accelerator
  --seed SEED           random seed (default: 1)
  --save_model          For saving the current Model
  --log_interval        how many batches to wait before logging training status
  --dry-run             quickly check a single pass
```
