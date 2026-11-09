from __future__ import print_function
import argparse, random, copy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision import transforms as T
from torch.optim.lr_scheduler import StepLR


class SiameseNetwork(nn.Module):
    """
        Siamese network for image similarity estimation.
        The network is composed of two identical networks, one for each input.
        The output of each network is concatenated and passed to a linear layer. 
        The output of the linear layer passed through a sigmoid function.
        `"FaceNet" <https://arxiv.org/pdf/1503.03832.pdf>`_ is a variant of the Siamese network.
        This implementation varies from FaceNet as we use the `ResNet-18` model from
        `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_ as our feature extractor.
        In addition, we aren't using `TripletLoss` as the MNIST dataset is simple, so `BCELoss` can do the trick.
    """
    # 孪生网络用于估计图片相似度。两个相同的子网络分别处理各自输入，
    # 将输出拼接后送入线性层，再经过 sigmoid。FaceNet 是孪生网络的一种变体。
    # 本实现使用 Deep Residual Learning for Image Recognition 中的 ResNet-18 作为特征提取器，
    # 因此与 FaceNet 不同。由于 MNIST 较简单，使用 BCELoss 即可，未使用 TripletLoss。
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        # get resnet model
        # 获取 ResNet 模型。
        self.resnet = torchvision.models.resnet18(weights=None)

        # over-write the first conv layer to be able to read MNIST images
        # 覆盖第一个卷积层，使其能够读取 MNIST 图片，
        # as resnet18 reads (3,x,x) where 3 is RGB channels
        # 因为 ResNet18 接收 (3, x, x)，其中 3 为 RGB 通道数，
        # whereas MNIST has (1,x,x) where 1 is a gray-scale channel
        # 而 MNIST 为 (1, x, x)，其中 1 为灰度通道数。
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.fc_in_features = self.resnet.fc.in_features
        
        # remove the last layer of resnet18 (linear layer which is before avgpool layer)
        # 移除 ResNet18 的最后一层；原注释将其描述为平均池化层之前的线性层。
        self.resnet = torch.nn.Sequential(*(list(self.resnet.children())[:-1]))

        # add linear layers to compare between the features of the two images
        # 添加线性层，比较两张图片的特征。
        self.fc = nn.Sequential(
            nn.Linear(self.fc_in_features * 2, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )

        self.sigmoid = nn.Sigmoid()

        # initialize the weights
        # 初始化权重。
        self.resnet.apply(self.init_weights)
        self.fc.apply(self.init_weights)
        
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward_once(self, x):
        output = self.resnet(x)
        output = output.view(output.size()[0], -1)
        return output

    def forward(self, input1, input2):
        # get two images' features
        # 获取两张图片的特征。
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        # concatenate both images' features
        # 拼接两张图片的特征。
        output = torch.cat((output1, output2), 1)

        # pass the concatenation to the linear layers
        # 将拼接结果传入线性层。
        output = self.fc(output)

        # pass the out of the linear layers to sigmoid layer
        # 将线性层输出传入 sigmoid 层。
        output = self.sigmoid(output)
        
        return output

class APP_MATCHER(Dataset):
    def __init__(self, root, train, download=False):
        super(APP_MATCHER, self).__init__()

        # get MNIST dataset
        # 获取 MNIST 数据集。
        self.dataset = datasets.MNIST(root, train=train, download=download)
        
        # as `self.dataset.data`'s shape is (Nx28x28), where N is the number of
        # self.dataset.data 的形状为 (N, 28, 28)，其中 N 是
        # examples in MNIST dataset, a single example has the dimensions of
        # MNIST 数据集的样本数量，单个样本的维度为
        # (28x28) for (WxH), where W and H are the width and the height of the image. 
        # (28, 28)，分别对应宽度 W 和高度 H。
        # However, every example should have (CxWxH) dimensions where C is the number 
        # 送入网络时，每个样本应具有 (C, W, H) 维度，C 为
        # of channels to be passed to the network. As MNIST contains gray-scale images, 
        # 通道数。由于 MNIST 是灰度图片，
        # we add an additional dimension to corresponds to the number of channels.
        # 需要额外添加一个通道维度。
        self.data = self.dataset.data.unsqueeze(1).clone()

        self.group_examples()

    def group_examples(self):
        """
            To ease the accessibility of data based on the class, we will use `group_examples` to group 
            examples based on class. 
            
            Every key in `grouped_examples` corresponds to a class in MNIST dataset. For every key in 
            `grouped_examples`, every value will conform to all of the indices for the MNIST 
            dataset examples that correspond to that key.
        """
        # 为便于按类别访问数据，使用 group_examples 对样本分组。
        # grouped_examples 的每个键对应 MNIST 的一个类别，
        # 对应值包含该类别所有样本在 MNIST 数据集中的索引。

        # get the targets from MNIST dataset
        # 获取 MNIST 数据集的目标标签。
        np_arr = np.array(self.dataset.targets.clone(), dtype=None, copy=None)
        
        # group examples based on class
        # 按类别对样本分组。
        self.grouped_examples = {}
        for i in range(0,10):
            self.grouped_examples[i] = np.where((np_arr==i))[0]
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, index):
        """
            For every example, we will select two images. There are two cases, 
            positive and negative examples. For positive examples, we will have two 
            images from the same class. For negative examples, we will have two images 
            from different classes.

            Given an index, if the index is even, we will pick the second image from the same class, 
            but it won't be the same image we chose for the first class. This is used to ensure the positive
            example isn't trivial as the network would easily distinguish the similarity between same images. However,
            if the network were given two different images from the same class, the network will need to learn 
            the similarity between two different images representing the same class. If the index is odd, we will 
            pick the second image from a different class than the first image.
        """
        # 每个样本选取两张图片：正例来自相同类别，负例来自不同类别。
        # 给定偶数索引时，第二张图片选自同一类别，但不能与第一张完全相同，
        # 避免任务过于简单，使网络学习同类不同图片之间的相似性。
        # 给定奇数索引时，第二张图片选自不同类别。

        # pick some random class for the first image
        # 为第一张图片随机选择一个类别。
        selected_class = random.randint(0, 9)

        # pick a random index for the first image in the grouped indices based of the label
        # 在该类别对应的索引组中随机选择
        # of the class
        # 第一张图片的索引。
        random_index_1 = random.randint(0, self.grouped_examples[selected_class].shape[0]-1)
        
        # pick the index to get the first image
        # 选择用于获取第一张图片的索引。
        index_1 = self.grouped_examples[selected_class][random_index_1]

        # get the first image
        # 获取第一张图片。
        image_1 = self.data[index_1].clone().float()

        # same class
        # 相同类别。
        if index % 2 == 0:
            # pick a random index for the second image
            # 为第二张图片随机选择一个索引。
            random_index_2 = random.randint(0, self.grouped_examples[selected_class].shape[0]-1)
            
            # ensure that the index of the second image isn't the same as the first image
            # 确保第二张图片与第一张图片的索引不同。
            while random_index_2 == random_index_1:
                random_index_2 = random.randint(0, self.grouped_examples[selected_class].shape[0]-1)
            
            # pick the index to get the second image
            # 选择用于获取第二张图片的索引。
            index_2 = self.grouped_examples[selected_class][random_index_2]

            # get the second image
            # 获取第二张图片。
            image_2 = self.data[index_2].clone().float()

            # set the label for this example to be positive (1)
            # 将此样本对的标签设为正例（1）。
            target = torch.tensor(1, dtype=torch.float)
        
        # different class
        # 不同类别。
        else:
            # pick a random class
            # 随机选择一个类别。
            other_selected_class = random.randint(0, 9)

            # ensure that the class of the second image isn't the same as the first image
            # 确保第二张图片的类别不同于第一张图片。
            while other_selected_class == selected_class:
                other_selected_class = random.randint(0, 9)

            
            # pick a random index for the second image in the grouped indices based of the label
            # 在该类别对应的索引组中随机选择第二张图片的索引。
            # of the class
            # 第一张图片的索引。
            random_index_2 = random.randint(0, self.grouped_examples[other_selected_class].shape[0]-1)

            # pick the index to get the second image
            # 选择用于获取第二张图片的索引。
            index_2 = self.grouped_examples[other_selected_class][random_index_2]

            # get the second image
            # 获取第二张图片。
            image_2 = self.data[index_2].clone().float()

            # set the label for this example to be negative (0)
            # 将此样本对的标签设为负例（0）。
            target = torch.tensor(0, dtype=torch.float)

        return image_1, image_2, target


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()

    # we aren't using `TripletLoss` as the MNIST dataset is simple, so `BCELoss` can do the trick.
    # 由于 MNIST 数据集较简单，使用 BCELoss 即可，因此未使用 TripletLoss。
    criterion = nn.BCELoss()

    for batch_idx, (images_1, images_2, targets) in enumerate(train_loader):
        images_1, images_2, targets = images_1.to(device), images_2.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(images_1, images_2).squeeze()
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(images_1), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    # we aren't using `TripletLoss` as the MNIST dataset is simple, so `BCELoss` can do the trick.
    # 由于 MNIST 数据集较简单，使用 BCELoss 即可，因此未使用 TripletLoss。
    criterion = nn.BCELoss()

    with torch.no_grad():
        for (images_1, images_2, targets) in test_loader:
            images_1, images_2, targets = images_1.to(device), images_2.to(device), targets.to(device)
            outputs = model(images_1, images_2).squeeze()
            test_loss += criterion(outputs, targets).sum().item()  # sum up batch loss
            # 累加各批次损失。
            pred = torch.where(outputs > 0.5, 1, 0)  # get the index of the max log-probability
            # 获取对数概率最大项的索引。
            correct += pred.eq(targets.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    # for the 1st epoch, the average loss is 0.0001 and the accuracy 97-98%
    # 原注释记录：默认设置下，第 1 轮的平均损失为 0.0001，准确率为 97%～98%；
    # using default settings. After completing the 10th epoch, the average
    # 完成第 10 轮后，平均
    # loss is 0.0000 and the accuracy 99.5-100% using default settings.
    # 损失为 0.0000，准确率为 99.5%～100%。
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    # 训练设置。
    parser = argparse.ArgumentParser(description='PyTorch Siamese network Example')
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
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    
    use_accel = not args.no_accel and torch.accelerator.is_available()

    torch.manual_seed(args.seed)


    if use_accel:
        device = torch.accelerator.current_accelerator()
    else:
        device = torch.device("cpu")
    
    print(f"Using device: {device}")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_accel:
        accel_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(accel_kwargs)
        test_kwargs.update(accel_kwargs)

    train_dataset = APP_MATCHER('../data', train=True, download=True)
    test_dataset = APP_MATCHER('../data', train=False)
    train_loader = torch.utils.data.DataLoader(train_dataset,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

    model = SiameseNetwork().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "siamese_network.pt")


if __name__ == '__main__':
    main()
