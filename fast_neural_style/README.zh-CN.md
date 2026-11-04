# 快速神经风格迁移 :city_sunrise: :rocket:

[English original](README.md)

> 中文说明；代码块中的命令、代码及程序输出保留原样。

本目录提供艺术风格迁移算法的 PyTorch 实现，将一张图片的内容与另一张图片的风格融合。例如，下图将门拱照片转换成彩色玻璃画风格。

模型采用论文 [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/abs/1603.08155) 中的方法，并结合[实例归一化（Instance Normalization）](https://arxiv.org/pdf/1607.08022.pdf)。本文示例所用的预训练模型可从[这里下载](https://www.dropbox.com/s/lrvwfehqdcxoza8/saved_models.zip?dl=0)。

<p align="center">
    <img src="images/style-images/mosaic.jpg" height="200px">
    <img src="images/content-images/amber.jpg" height="200px">
    <img src="images/output-images/amber-mosaic.jpg" height="440px">
</p>

## 环境依赖

程序使用 Python 编写，依赖 [PyTorch](http://pytorch.org/) 和 [SciPy](https://www.scipy.org)。GPU 不是必需的，但可以明显加速计算，尤其是训练新模型时。使用已保存模型时，普通尺寸的图片可以在笔记本或台式机上进行风格迁移。

## 使用方法

对图片进行风格迁移：

```
python neural_style/neural_style.py eval --content-image </path/to/content/image> --model </path/to/saved/model> --output-image </path/to/output/image> --accel
```

- `--content-image`：需要转换风格的内容图片路径。
- `--model`：用于转换图片的已保存模型，例如 `mosaic.pth`。
- `--output-image`：输出图片的保存路径。
- `--content-scale`：内存不足时，按该系数缩小内容图片。例如设为 2 时，图片高度和宽度各减半。
- `--accel`：使用加速设备。

训练模型：

```bash
python neural_style/neural_style.py train --dataset </path/to/train-dataset> --style-image </path/to/style/image> --save-model-dir </path/to/save-model/folder> --epochs 2 --accel
```

主要命令行参数如下：

- `--dataset`：训练数据集路径，该目录应包含一个存放全部训练图片的子目录。原作者使用 COCO 2014 训练图片集，约 80,000 张、13 GB（[下载](https://cocodataset.org/#download)）。
- `--style-image`：风格图片路径。
- `--save-model-dir`：训练后模型的保存目录。
- `--accel`：使用加速设备。

指定 `--accel` 后，PyTorch 会搜索可用硬件加速设备并尝试使用。本示例可在 CUDA、MPS、XPU 设备上运行。

其他命令行参数见 `neural_style/neural_style.py`。训练新模型时，可能需要调整 `--content-weight` 和 `--style-weight`。上图马赛克风格模型使用 `--content-weight 1e5` 和 `--style-weight 1e10`。其余三个模型使用相近数量级的权重，其中 `--style-weight` 略有不同，设为 `5e10` 或 `1e11`。

## 模型

下列示例的模型可从[这里下载](https://www.dropbox.com/s/lrvwfehqdcxoza8/saved_models.zip?dl=0)，也可以运行脚本 `download_saved_models.py` 下载。

<div align='center'>
  <img src='images/content-images/amber.jpg' height="174px">
</div>

<div align='center'>
  <img src='images/style-images/mosaic.jpg' height="174px">
  <img src='images/output-images/amber-mosaic.jpg' height="174px">
  <img src='images/output-images/amber-candy.jpg' height="174px">
  <img src='images/style-images/candy.jpg' height="174px">
  <br>
  <img src='images/style-images/rain-princess-cropped.jpg' height="174px">
  <img src='images/output-images/amber-rain-princess.jpg' height="174px">
  <img src='images/output-images/amber-udnie.jpg' height="174px">
  <img src='images/style-images/udnie.jpg' height="174px">
</div>
