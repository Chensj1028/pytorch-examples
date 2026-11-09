# fast-neural-style :city_sunrise: :rocket:

快速神经风格迁移 :city_sunrise: :rocket:

This repository contains a pytorch implementation of an algorithm for artistic style transfer. The algorithm can be used to mix the content of an image with the style of another image. For example, here is a photograph of a door arch rendered in the style of a stained glass painting.

本目录提供艺术风格迁移算法的 PyTorch 实现，将一张图片的内容与另一张图片的风格融合。例如，下图将门拱照片转换成彩色玻璃画风格。

The model uses the method described in [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/abs/1603.08155) along with [Instance Normalization](https://arxiv.org/pdf/1607.08022.pdf). The saved-models for examples shown in the README can be downloaded from [here](https://www.dropbox.com/s/lrvwfehqdcxoza8/saved_models.zip?dl=0).

模型采用论文 [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/abs/1603.08155) 中的方法，并结合[实例归一化（Instance Normalization）](https://arxiv.org/pdf/1607.08022.pdf)。本文示例所用的预训练模型可从[这里下载](https://www.dropbox.com/s/lrvwfehqdcxoza8/saved_models.zip?dl=0)。

<p align="center">
    <img src="images/style-images/mosaic.jpg" height="200px">
    <img src="images/content-images/amber.jpg" height="200px">
    <img src="images/output-images/amber-mosaic.jpg" height="440px">
</p>

## Requirements

环境依赖

The program is written in Python, and uses [pytorch](http://pytorch.org/), [scipy](https://www.scipy.org). A GPU is not necessary, but can provide a significant speed up especially for training a new model. Regular sized images can be styled on a laptop or desktop using saved models.

程序使用 Python 编写，依赖 [PyTorch](http://pytorch.org/) 和 [SciPy](https://www.scipy.org)。GPU 不是必需的，但可以明显加速计算，尤其是训练新模型时。使用已保存模型时，普通尺寸的图片可以在笔记本或台式机上进行风格迁移。

## Usage

使用方法

Stylize image

对图片进行风格迁移：

```
python neural_style/neural_style.py eval --content-image </path/to/content/image> --model </path/to/saved/model> --output-image </path/to/output/image> --accel
```

- `--content-image`: path to content image you want to stylize.

  `--content-image`：需要转换风格的内容图片路径。

- `--model`: saved model to be used for stylizing the image (eg: `mosaic.pth`)

  `--model`：用于转换图片的已保存模型，例如 `mosaic.pth`。

- `--output-image`: path for saving the output image.

  `--output-image`：输出图片的保存路径。

- `--content-scale`: factor for scaling down the content image if memory is an issue (eg: value of 2 will halve the height and width of content-image)

  `--content-scale`：内存不足时，按该系数缩小内容图片。例如设为 2 时，图片高度和宽度各减半。

- `--accel`: use accelerator

  `--accel`：使用加速设备。

Train model

训练模型：

```bash
python neural_style/neural_style.py train --dataset </path/to/train-dataset> --style-image </path/to/style/image> --save-model-dir </path/to/save-model/folder> --epochs 2 --accel
```

There are several command line arguments, the important ones are listed below

主要命令行参数如下：

- `--dataset`: path to training dataset, the path should point to a folder containing another folder with all the training images. I used COCO 2014 Training images dataset [80K/13GB] [(download)](https://cocodataset.org/#download).

  `--dataset`：训练数据集路径，该目录应包含一个存放全部训练图片的子目录。原作者使用 COCO 2014 训练图片集，约 80,000 张、13 GB（[下载](https://cocodataset.org/#download)）。

- `--style-image`: path to style-image.

  `--style-image`：风格图片路径。

- `--save-model-dir`: path to folder where trained model will be saved.

  `--save-model-dir`：训练后模型的保存目录。

- `--accel`: use accelerator.

  `--accel`：使用加速设备。

If `--accel` argument is given, pytorch will search for available hardware acceleration device and attempt to use it. This example is known to work on CUDA, MPS and XPU devices.

指定 `--accel` 后，PyTorch 会搜索可用硬件加速设备并尝试使用。本示例可在 CUDA、MPS、XPU 设备上运行。

Refer to `neural_style/neural_style.py` for other command line arguments. For training new models you might have to tune the values of `--content-weight` and `--style-weight`. The mosaic style model shown above was trained with `--content-weight 1e5` and `--style-weight 1e10`. The remaining 3 models were also trained with similar order of weight parameters with slight variation in the `--style-weight` (`5e10` or `1e11`).

其他命令行参数见 `neural_style/neural_style.py`。训练新模型时，可能需要调整 `--content-weight` 和 `--style-weight`。上图马赛克风格模型使用 `--content-weight 1e5` 和 `--style-weight 1e10`。其余三个模型使用相近数量级的权重，其中 `--style-weight` 略有不同，设为 `5e10` 或 `1e11`。

## Models

模型

Models for the examples shown below can be downloaded from [here](https://www.dropbox.com/s/lrvwfehqdcxoza8/saved_models.zip?dl=0) or by running the script `download_saved_models.py`.

下列示例的模型可从[这里下载](https://www.dropbox.com/s/lrvwfehqdcxoza8/saved_models.zip?dl=0)，也可以运行脚本 `download_saved_models.py` 下载。

<div align='center'>
  <img src='images/content-images/amber.jpg' height="174px">		
</div>

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
