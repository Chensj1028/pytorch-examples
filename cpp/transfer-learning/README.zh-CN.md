# 使用 LibTorch 和 OpenCV 在猫狗数据集上进行迁移学习

[English original](README.md)

> 中文说明；代码块中的命令、代码及程序输出保留原样。

本例通过 PyTorch C++ API，在 Dogs vs Cats 数据集上进行迁移学习。

## 使用方法

**训练：**

1. 移除预训练 `ResNet18` 模型的最后一层，并转成 `torch.jit` 模块：`python3 convert.py`。
2. 创建并进入构建目录：`mkdir build && cd build`。
3. 执行 `cmake -DCMAKE_PREFIX_PATH=/absolute/path/to/libtorch ..`。
4. 执行 `make`。
5. 运行训练代码：`./example <path_to_scripting_model>`。

**预测：**

1. 执行 `cd build`。
2. 命令格式：`./classify <path_image> <path_to_resnet18_model_without_fc_layer> <model_linear_trained>`；示例：`./classify <path_image> ../resnet18_without_last_layer.pt model_linear.pt`。

使用 LibTorch 进行迁移学习的详细介绍见[配套博客](https://krshrimali.github.io/Applying-Transfer-Learning-Dogs-Cats/)。
