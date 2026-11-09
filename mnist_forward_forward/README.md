# Basic Forward-Forward Example

Forward-Forward 基础示例

This example implements the paper [The Forward-Forward Algorithm: Some Preliminary Investigations](https://arxiv.org/abs/2212.13345) by Geoffrey Hinton.

本示例实现 Geoffrey Hinton 的论文 [The Forward-Forward Algorithm: Some Preliminary Investigations](https://arxiv.org/abs/2212.13345)。

the aim of this paper is to introduce a new learning procedure for neural networks. the forward and backward passes of backpropagation by two forward passes.

论文提出一种新的神经网络学习方法：用两次前向传播，替代反向传播训练中的前向与反向过程。

```bash
pip install -r requirements.txt
python main.py
```

The main.py script accepts the following arguments:

`main.py` 脚本接受以下参数：

```bash
optional arguments:
  -h, --help            show this help message and exit
  --epochs EPOCHS       number of epochs to train (default: 1000)
  --lr LR               learning rate (default: 0.03)
  --no_accel            disables accelerator
  --seed SEED           random seed (default: 1)
  --save_model          For saving the current Model
  --train_size TRAIN_SIZE
                        size of training set
  --threshold THRESHOLD
                        threshold for training
  --test_size TEST_SIZE
                        size of test set
  --save-model          For Saving the current Model
  --log-interval LOG_INTERVAL
                        logging training status interval
```
