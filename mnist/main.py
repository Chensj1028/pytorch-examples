# 阅读顺序：文件底部的入口 → main() → Net → train() → test()。
# 总体流程：准备图片和正确答案 → 创建模型 → 训练 → 测试 → 保存参数。
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR


# 继承 nn.Module 后，PyTorch 可以自动管理网络中需要学习的参数。
class Net(nn.Module):
    def __init__(self):
        # __init__ 定义网络的各层，先初始化父类；forward 决定数据通过各层的顺序。
        super(Net, self).__init__()
        # 四个参数：输入通道数、输出通道数、卷积核大小、步长。
        # 灰度图有 1 个通道；32 个 3×3 卷积核每次移动 1 像素，提取局部特征。
        # 默认不补边，因此这次卷积后，图片的宽、高各减少 2。
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        # 接收上一层的 32 个特征通道，再提取出 64 个通道。
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        # Dropout 在训练时按给定概率随机将元素置零，减少对个别特征的依赖。
        # 0.25 和 0.5 是置零概率；model.eval() 会关闭这种随机丢弃行为。
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        # Linear 是全连接层，两个参数是输入、输出的特征数量。
        # 9216 = 64 × 12 × 12，来自卷积与池化后的形状，详见 forward。
        self.fc1 = nn.Linear(9216, 128)
        # 将 128 个特征变成 10 个类别分数，下标 0～9 对应数字 0～9。
        self.fc2 = nn.Linear(128, 10)

    # 调用 model(data) 时会执行 forward，完成一次前向传播。
    # 张量（Tensor）可以先理解为多维数字数组；B 表示一批图片的数量。
    def forward(self, x):
        # 输入 [B, 1, 28, 28]：批大小、通道数、高、宽。
        # 第一次卷积后变成 [B, 32, 26, 26]。
        x = self.conv1(x)
        # ReLU 把负数变成 0、正数保持不变，引入非线性，不改变张量形状。
        x = F.relu(x)
        # 第二次卷积后：[B, 64, 24, 24]。
        x = self.conv2(x)
        # ReLU 把负数变成 0、正数保持不变，引入非线性，不改变张量形状。
        x = F.relu(x)
        # 每个 2×2 区域保留最大值，宽、高各减半 → [B, 64, 12, 12]。
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        # 保留第 0 维的批大小，把后面的维度展平 → [B, 9216]。
        # 每张图片的特征变成一排数字，供全连接层使用。
        x = torch.flatten(x, 1)
        # 第一个全连接层将特征数量从 9216 压缩到 128 → [B, 128]。
        x = self.fc1(x)
        # ReLU 把负数变成 0、正数保持不变，引入非线性，不改变张量形状。
        x = F.relu(x)
        x = self.dropout2(x)
        # 最后输出 [B, 10]：每张图片对应数字 0～9 的 10 个分数。
        x = self.fc2(x)
        # 沿类别维（dim=1）计算对数概率，供训练中的 nll_loss 使用。
        # 这是概率的对数，不能直接当作百分比；最大值对应模型预测的类别。
        output = F.log_softmax(x, dim=1)
        return output


# 每遍历一次完整训练集叫一轮（epoch）；这个函数执行一轮训练。
def train(args, model, device, train_loader, optimizer, epoch):
    # 切换到训练模式，让 Dropout 按训练方式工作；这行本身不会更新参数。
    model.train()
    # 每次取一批图片 data 和正确数字 target（标签）；batch_idx 是批次编号。
    # 默认批大小 64：data 通常是 [64, 1, 28, 28]，target 是 [64]。
    # 最后一批可能不足 64 张。
    for batch_idx, (data, target) in enumerate(train_loader):
        # 图片、标签和模型需要在同一个计算设备上，例如都位于 CPU。
        data, target = data.to(device), target.to(device)
        # ① 清除上一批的梯度。PyTorch 默认会累加梯度，所以每批训练前要清除。
        # 这里只清除梯度，不会清除模型已经学到的参数值。
        optimizer.zero_grad()
        # ② 前向传播：让模型预测，得到 [批大小, 10] 的对数概率。
        output = model(data)
        # ③ 计算损失：比较预测和正确标签，得到表示错误程度的数。
        # nll_loss 默认计算这一批的平均损失；给正确类别的概率越大，损失越小。
        loss = F.nll_loss(output, target)
        # ④ 反向传播：计算损失对各个可学习参数的梯度，此时还没有修改参数。
        # 梯度描述参数变化对损失的影响，供优化器决定怎样调整参数。
        loss.backward()
        # ⑤ 更新参数：优化器根据梯度调整模型参数，这一步真正改变了模型。
        optimizer.step()
        # 每隔指定批次数打印一次进度；单个批次的损失有起伏是正常的。
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            # 快速检查模式在第一次打印后结束本轮训练，随后仍会执行测试。
            if args.dry_run:
                break


# 使用测试集评估模型，不在测试数据上执行参数更新。
def test(model, device, test_loader):
    # 切换到评估模式，例如关闭 Dropout 的随机丢弃行为。
    model.eval()
    test_loss = 0
    correct = 0
    # 测试不需要求梯度，关闭梯度记录可节省内存和计算。
    # no_grad 与 eval 的作用不同，测试时通常需要一起使用。
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # 累加本批所有样本的损失；item() 将单个数值的张量转成 Python 数值。
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            # 沿类别维找最大值的下标，作为预测数字；结果形状是 [批大小, 1]。
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            # 调整标签形状，与预测逐个比较，再累加识别正确的数量。
            correct += pred.eq(target.view_as(pred)).sum().item()

    # 总损失除以测试图片数，得到平均每张图片的损失。
    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


# main 负责组织整个流程，第一次阅读可以先浏览这个函数的后半部分。
def main():
    # 命令行参数：batch-size 是批大小，epochs 是轮数，lr 是学习率。
    # 源码默认 14 轮；run_mnist.cmd 通过 --epochs 1 改为训练 1 轮。
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-accel', action='store_true',
                        help='disables accelerator')
    parser.add_argument('--dry-run', action='store_true',
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', 
                        help='For Saving the current Model')
    args = parser.parse_args()

    # 未指定 --no-accel 且存在可用加速设备时，才使用加速设备。
    use_accel = not args.no_accel and torch.accelerator.is_available()

    # 固定随机种子以便比较实验；不同设备、环境仍可能存在结果差异。
    torch.manual_seed(args.seed)

    if use_accel:
        device = torch.accelerator.current_accelerator()
    else:
        device = torch.device("cpu")

    # 先准备 DataLoader 配置；后面的 **字典 将配置展开为关键字参数。
    # 只有下面的加速设备分支设置了 shuffle=True，CPU 分支默认按顺序加载。
    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_accel:
        accel_kwargs = {'num_workers': 1,
                        'persistent_workers': True,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(accel_kwargs)
        test_kwargs.update(accel_kwargs)

    # Compose 按顺序组合预处理操作。
    transform=transforms.Compose([
        # 转为 [1, 28, 28] 的张量，像素值从 0～255 缩放到 0～1。
        transforms.ToTensor(),
        # 标准化：(像素值 - 均值) / 标准差，常数来自 MNIST 训练集的统计。
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    # 获取 60,000 张训练图片；缺失时自动下载，下载同时准备训练集和测试集。
    # ../data 相对于工作目录；run_mnist.cmd 会先进入 mnist 目录。
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
    # train=False 获取 10,000 张测试图片，用于评估，不参与参数更新。
    dataset2 = datasets.MNIST('../data', train=False,
                       transform=transform)
    # Dataset 管理单个样本，DataLoader 负责将样本组织成一批批的数据。
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    # 每次运行都会创建一个全新模型，再将它的参数放到所选设备上。
    model = Net().to(device)
    # model.parameters() 提供需要学习的参数；优化器根据梯度更新这些参数。
    # lr 是学习率，用于控制更新的尺度。
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    # 每完成一轮，将学习率乘以 gamma（默认 0.7），供下一轮使用。
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    # range 不包含右端点，因此加 1 后轮数编号是 1 到 args.epochs。
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    if args.save_model:
        # 保存模型参数；以后要先创建相同的 Net，再加载这些参数。
        # 这行不保存网络结构，同名文件会被覆盖；本脚本没有实现继续训练。
        torch.save(model.state_dict(), "mnist_cnn.pt")


# 直接运行本文件时执行 main()；被其他文件 import 时不自动开始训练。
if __name__ == '__main__':
    main()
