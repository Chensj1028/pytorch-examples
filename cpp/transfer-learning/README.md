# Transfer Learning on Dogs vs Cats Dataset using Libtorch and OpenCV

使用 LibTorch 和 OpenCV 在猫狗数据集上进行迁移学习

Transfer Learning on Dogs vs Cats dataset using PyTorch C++ API.

本例通过 PyTorch C++ API，在 Dogs vs Cats 数据集上进行迁移学习。

## Usage

使用方法

For **training**:

**训练：**

1. Remove final layer of `ResNet18` pre-trained model and convert to `torch.jit` module: `python3 convert.py`.

   移除预训练 `ResNet18` 模型的最后一层，并转成 `torch.jit` 模块：`python3 convert.py`。

2. Create build directory: `mkdir build && cd build`

   创建并进入构建目录：`mkdir build && cd build`。

3. `cmake -DCMAKE_PREFIX_PATH=/absolute/path/to/libtorch ..`

   执行 `cmake -DCMAKE_PREFIX_PATH=/absolute/path/to/libtorch ..`。

4. `make`

   执行 `make`。

5. Run training code: `./example <path_to_scripting_model>`

   运行训练代码：`./example <path_to_scripting_model>`。

For **prediction**:

**预测：**

1. `cd build`

   执行 `cd build`。

2. `./classify <path_image> <path_to_resnet18_model_without_fc_layer> <model_linear_trained>` : `./classify <path_image> ../resnet18_without_last_layer.pt model_linear.pt`

   命令格式：`./classify <path_image> <path_to_resnet18_model_without_fc_layer> <model_linear_trained>`；示例：`./classify <path_image> ../resnet18_without_last_layer.pt model_linear.pt`。

Detailed blog on applying Transfer Learning using Libtorch: https://krshrimali.github.io/Applying-Transfer-Learning-Dogs-Cats/.

使用 LibTorch 进行迁移学习的详细介绍见[配套博客](https://krshrimali.github.io/Applying-Transfer-Learning-Dogs-Cats/)。
