# ImageNet training in PyTorch

使用 PyTorch 在 ImageNet 上训练

This implements training of popular model architectures, such as ResNet, AlexNet, and VGG on the ImageNet dataset.

本示例在 ImageNet 数据集上训练 ResNet、AlexNet、VGG 等常见模型架构。

## Requirements

环境与数据准备

- Install PyTorch ([pytorch.org](http://pytorch.org))

  安装 PyTorch（[pytorch.org](http://pytorch.org)）。

- `pip install -r requirements.txt`

  执行 `pip install -r requirements.txt`。

- Download the ImageNet dataset from http://www.image-net.org/

  从 http://www.image-net.org/ 下载 ImageNet 数据集。

  - Then, move and extract the training and validation images to labeled subfolders, using [the following shell script](extract_ILSVRC.sh)

    使用[解压脚本](extract_ILSVRC.sh)，将训练和验证图片移动、解压到按类别组织的子目录中。

## Training

训练

To train a model, run `main.py` with the desired model architecture and the path to the ImageNet dataset:

运行 `main.py`，指定模型架构和 ImageNet 数据集路径：

```bash
python main.py -a resnet18 [imagenet-folder with train and val folders]
```

The default learning rate schedule starts at 0.1 and decays by a factor of 10 every 30 epochs. This is appropriate for ResNet and models with batch normalization, but too high for AlexNet and VGG. Use 0.01 as the initial learning rate for AlexNet or VGG:

默认学习率从 0.1 开始，每训练 30 轮缩小为原来的十分之一。这适用于 ResNet 和使用批归一化的模型，但对 AlexNet、VGG 来说偏大。训练 AlexNet 或 VGG 时，初始学习率应使用 0.01：

```bash
python main.py -a alexnet --lr 0.01 [imagenet-folder with train and val folders]
```

## Use Dummy Data

使用模拟数据

ImageNet dataset is large and time-consuming to download. To get started quickly, run `main.py` using dummy data by "--dummy". It's also useful for training speed benchmark. Note that the loss or accuracy is useless in this case.

ImageNet 数据集体积较大，下载需要时间。可以给 `main.py` 添加 `--dummy`，使用模拟数据快速体验，也可用来测试训练速度。此时损失和准确率没有实际评价意义。

```bash
python main.py -a resnet18 --dummy
```

## Multi-processing Distributed Data Parallel Training

多进程分布式数据并行训练

If running on CUDA, you should always use the NCCL backend for multi-processing distributed training since it currently provides the best distributed training performance.

按照原版说明，在 CUDA 上进行多进程分布式训练时，应使用 NCCL 后端以获得较好的分布式训练性能。

For XPU multiprocessing is not supported as of PyTorch 2.6.

原版同时注明：截至 PyTorch 2.6，XPU 尚不支持这里的多进程方式。

### Single node, multiple GPUs:

单节点、多 GPU

```bash
python main.py -a resnet50 --dist-url 'tcp://127.0.0.1:FREEPORT' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 [imagenet-folder with train and val folders]
```

### Multiple nodes:

多节点

Node 0:

节点 0：

```bash
python main.py -a resnet50 --dist-url 'tcp://IP_OF_NODE0:FREEPORT' --dist-backend 'nccl' --multiprocessing-distributed --world-size 2 --rank 0 [imagenet-folder with train and val folders]
```

Node 1:

节点 1：

```bash
python main.py -a resnet50 --dist-url 'tcp://IP_OF_NODE0:FREEPORT' --dist-backend 'nccl' --multiprocessing-distributed --world-size 2 --rank 1 [imagenet-folder with train and val folders]
```

## Usage

命令行用法

```bash
usage: main.py [-h] [-a ARCH] [-j N] [--epochs N] [--start-epoch N] [-b N] [--lr LR] [--momentum M] [--wd W] [-p N] [--resume PATH] [-e] [--pretrained] [--world-size WORLD_SIZE] [--rank RANK]
               [--dist-url DIST_URL] [--dist-backend DIST_BACKEND] [--seed SEED] [--gpu GPU] [--no-accel][--multiprocessing-distributed] [--dummy]
               [DIR]

PyTorch ImageNet Training

positional arguments:
  DIR                   path to dataset (default: imagenet)

optional arguments:
  -h, --help            show this help message and exit
  -a ARCH, --arch ARCH  model architecture: alexnet | convnext_base | convnext_large | convnext_small | convnext_tiny | densenet121 | densenet161 | densenet169 | densenet201 | efficientnet_b0 |
                        efficientnet_b1 | efficientnet_b2 | efficientnet_b3 | efficientnet_b4 | efficientnet_b5 | efficientnet_b6 | efficientnet_b7 | googlenet | inception_v3 | mnasnet0_5 | mnasnet0_75 |
                        mnasnet1_0 | mnasnet1_3 | mobilenet_v2 | mobilenet_v3_large | mobilenet_v3_small | regnet_x_16gf | regnet_x_1_6gf | regnet_x_32gf | regnet_x_3_2gf | regnet_x_400mf | regnet_x_800mf |
                        regnet_x_8gf | regnet_y_128gf | regnet_y_16gf | regnet_y_1_6gf | regnet_y_32gf | regnet_y_3_2gf | regnet_y_400mf | regnet_y_800mf | regnet_y_8gf | resnet101 | resnet152 | resnet18 |
                        resnet34 | resnet50 | resnext101_32x8d | resnext50_32x4d | shufflenet_v2_x0_5 | shufflenet_v2_x1_0 | shufflenet_v2_x1_5 | shufflenet_v2_x2_0 | squeezenet1_0 | squeezenet1_1 | vgg11 |
                        vgg11_bn | vgg13 | vgg13_bn | vgg16 | vgg16_bn | vgg19 | vgg19_bn | vit_b_16 | vit_b_32 | vit_l_16 | vit_l_32 | wide_resnet101_2 | wide_resnet50_2 (default: resnet18)
  -j N, --workers N     number of data loading workers (default: 4)
  --epochs N            number of total epochs to run
  --start-epoch N       manual epoch number (useful on restarts)
  -b N, --batch-size N  mini-batch size (default: 256), this is the total batch size of all GPUs on the current node when using Data Parallel or Distributed Data Parallel
  --lr LR, --learning-rate LR
                        initial learning rate
  --momentum M          momentum
  --wd W, --weight-decay W
                        weight decay (default: 1e-4)
  -p N, --print-freq N  print frequency (default: 10)
  --resume PATH         path to latest checkpoint (default: none)
  -e, --evaluate        evaluate model on validation set
  --pretrained          use pre-trained model
  --world-size WORLD_SIZE
                        number of nodes for distributed training
  --rank RANK           node rank for distributed training
  --dist-url DIST_URL   url used to set up distributed training
  --dist-backend DIST_BACKEND
                        distributed backend
  --seed SEED           seed for initializing training.
  --gpu GPU             GPU id to use.
  --no-accel            disables accelerator
  --multiprocessing-distributed
                        Use multi-processing distributed training to launch N processes per node, which has N GPUs. This is the fastest way to use PyTorch for either single node or multi node data parallel
                        training
  --dummy               use fake data to benchmark

```
