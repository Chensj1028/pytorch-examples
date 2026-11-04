# MNIST Hogwild 多进程训练示例

[English original](README.md)

> 中文说明；代码块中的命令、代码及程序输出保留原样。

```bash
pip install -r requirements.txt
python main.py
```

`main.py` 脚本接受以下参数：

```bash
optional arguments:
  -h, --help            show this help message and exit
  --batch-size          input batch_size for training (default: 64)
  --test-batch-size     input batch size for testing (default: 1000)
  --epochs EPOCHS       number of epochs to train (default: 10)
  --lr LR               learning rate (default: 0.01)
  --momentum            SGD momentum (default: 0.5)
  --seed SEED           random seed (default: 1)
  --save_model          save the trained model to state_dict
  --log-interval        how many batches to wait before logging training status (default: 10)
  --num-processes       how many training processes to use (default: 2)
  --cuda                enables CUDA training
  --mps                 enables macos GPU training
  --dry-run             quickly check a single pass
```
