# 变分自编码器（VAE）基础示例

[English original](README.md)

> 中文说明；代码块中的命令、代码及程序输出保留原样。

本示例改进实现了 Kingma 和 Welling 的论文 [Auto-Encoding Variational Bayes](http://arxiv.org/abs/1312.6114)。
相比原实现，它使用 ReLU 和 Adam 优化器，替换 sigmoid 和 Adagrad，使网络更快收敛。

### 使用方法

安装所需依赖：

```bash
pip install -r requirements.txt
```

运行示例：

```bash
python main.py
```

检测到可用硬件加速设备时，示例会使用该设备；否则使用 CPU。

如需强制使用 CPU，添加 `--no-accel` 参数：

```bash
python main.py --no-accel
```

`main.py` 脚本接受以下可选参数：

```bash
--batch-size            input batch size for training (default: 128)
--epochs                number of epochs to train (default: 10)
--no-accel              disables accelerator
--seed                  random seed (default: 1)
--log-interval	        how many batches to wait before logging training status
```
