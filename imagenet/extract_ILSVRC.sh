#!/bin/bash
#
# script to extract ImageNet dataset
# 解压 ImageNet 数据集的脚本。
# ILSVRC2012_img_train.tar (about 138 GB)
# ILSVRC2012_img_train.tar，约 138 GB。
# ILSVRC2012_img_val.tar (about 6.3 GB)
# ILSVRC2012_img_val.tar，约 6.3 GB。
# make sure ILSVRC2012_img_train.tar & ILSVRC2012_img_val.tar in your current directory
# 确保当前目录中有 ILSVRC2012_img_train.tar 和 ILSVRC2012_img_val.tar。
#
#  Adapted from:
# 改编来源：
#  https://github.com/facebook/fb.resnet.torch/blob/master/INSTALL.md
#  https://gist.github.com/BIGBALLON/8a71d225eff18d88e469e6ea9b39cef4
# 
#  imagenet/train/
#  ├── n01440764
#  │   ├── n01440764_10026.JPEG
#  │   ├── n01440764_10027.JPEG
#  │   ├── ......
#  ├── ......
#  imagenet/val/
#  ├── n01440764
#  │   ├── ILSVRC2012_val_00000293.JPEG
#  │   ├── ILSVRC2012_val_00002138.JPEG
#  │   ├── ......
#  ├── ......
#
#
# Make imagnet directory
# 创建 imagenet 目录。
#
mkdir imagenet
#
# Extract the training data:
# 解压训练数据：
#
# Create train directory; move .tar file; change directory
# 创建 train 目录，移动 tar 文件，再进入该目录。
mkdir imagenet/train && mv ILSVRC2012_img_train.tar imagenet/train/ && cd imagenet/train
# Extract training set; remove compressed file
# 解压训练集，删除压缩文件。
tar -xvf ILSVRC2012_img_train.tar && rm -f ILSVRC2012_img_train.tar
#
# At this stage imagenet/train will contain 1000 compressed .tar files, one for each category
# 此时 imagenet/train 中包含 1000 个 tar 压缩文件，每个类别一个。
#
# For each .tar file: 
# 对每个 tar 文件：
#   1. create directory with same name as .tar file
# 1．创建与 tar 文件同名的目录。
#   2. extract and copy contents of .tar file into directory
# 2．将 tar 文件的内容解压到该目录。
#   3. remove .tar file
# 3．删除 tar 文件。
find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done
#
# This results in a training directory like so:
# 得到如下训练目录结构：
#
#  imagenet/train/
#  ├── n01440764
#  │   ├── n01440764_10026.JPEG
#  │   ├── n01440764_10027.JPEG
#  │   ├── ......
#  ├── ......
#
# Change back to original directory
# 返回原目录。
cd ../..
#
# Extract the validation data and move images to subfolders:
# 解压验证数据，并将图片移动到各子目录：
#
# Create validation directory; move .tar file; change directory; extract validation .tar; remove compressed file
# 创建验证目录，移动 tar 文件，进入目录，解压并删除压缩文件。
mkdir imagenet/val && mv ILSVRC2012_img_val.tar imagenet/val/ && cd imagenet/val && tar -xvf ILSVRC2012_img_val.tar && rm -f ILSVRC2012_img_val.tar
# get script from soumith and run; this script creates all class directories and moves images into corresponding directories
# 获取 soumith 的脚本并运行；该脚本创建各类别目录，并将图片移到相应目录。
wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash
#
# This results in a validation directory like so:
# 得到如下验证目录结构：
#
#  imagenet/val/
#  ├── n01440764
#  │   ├── ILSVRC2012_val_00000293.JPEG
#  │   ├── ILSVRC2012_val_00002138.JPEG
#  │   ├── ......
#  ├── ......
#
#
# Check total files after extract
# 检查解压后的文件总数。
#
#  $ find train/ -name "*.JPEG" | wc -l
#  1281167
#  $ find val/ -name "*.JPEG" | wc -l
#  50000
#
