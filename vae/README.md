# Basic VAE Example

变分自编码器（VAE）基础示例

This is an improved implementation of the paper [Auto-Encoding Variational Bayes](http://arxiv.org/abs/1312.6114) by Kingma and Welling.
It uses ReLUs and the adam optimizer, instead of sigmoids and adagrad. These changes make the network converge much faster.

本示例改进实现了 Kingma 和 Welling 的论文 [Auto-Encoding Variational Bayes](http://arxiv.org/abs/1312.6114)。
相比原实现，它使用 ReLU 和 Adam 优化器，替换 sigmoid 和 Adagrad，使网络更快收敛。

### Usage

使用方法

Install the required dependencies:

安装所需依赖：

```bash
pip install -r requirements.txt
```

To run the example, execute:

运行示例：

```bash
python main.py
```

If a hardware accelerator device is detected, the example will execute on the accelerator; otherwise, it will run on the CPU.

检测到可用硬件加速设备时，示例会使用该设备；否则使用 CPU。

To force execution on the CPU, use `--no-accel` command line argument:

如需强制使用 CPU，添加 `--no-accel` 参数：

```bash
python main.py --no-accel
```

The main.py script accepts the following optional arguments:

`main.py` 脚本接受以下可选参数：

```bash
--batch-size            input batch size for training (default: 128)
--epochs                number of epochs to train (default: 10)
--no-accel              disables accelerator
--seed                  random seed (default: 1)
--log-interval	        how many batches to wait before logging training status
```
