# Installing OpenCV 

安装 OpenCV

## Linux with Package Manager

Linux：通过包管理器安装

### Arch Linux

Arch Linux 系统

```shell
pacman -Syu base-devel opencv
```

### Fedora

Fedora 系统

```shell
sudo dnf install opencv opencv-dev
```

## Linux From Source

Linux：从源码构建

Required Packages:

必需软件包：

```shell
sudo apt-get install build-essential cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev
```

Optional Packages:

可选软件包：

```shell
sudo apt-get install python-dev python-numpy libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libjasper-dev libdc1394-22-dev
```

Building from Source:

从源码构建：

```shell
git clone https://github.com/opencv/opencv.git
git clone https://github.com/opencv/opencv_contrib.git

cd opencv && mkdir build && cd build
cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/usr/local ..
make -j8 # runs 8 jobs in parallel
# 并行运行 8 个任务。
sudo make install
```

## Windows

Windows 系统

You can download the pre-built libraries from [OpenCV releases](https://github.com/opencv/opencv/releases) and install them easily.

可以从 [OpenCV 发布页](https://github.com/opencv/opencv/releases) 下载预编译库并安装。
