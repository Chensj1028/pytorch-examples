#!/usr/bin/env python
from __future__ import print_function
from itertools import count

import torch
import torch.nn.functional as F

POLY_DEGREE = 4
W_target = torch.randn(POLY_DEGREE, 1) * 5
b_target = torch.randn(1) * 5


def make_features(x):
    """Builds features i.e. a matrix with columns [x, x^2, x^3, x^4]."""
    # 构建特征矩阵，各列为 [x, x^2, x^3, x^4]。
    x = x.unsqueeze(1)
    return torch.cat([x ** i for i in range(1, POLY_DEGREE+1)], 1)


def f(x):
    """Approximated function."""
    # 拟合得到的函数。
    return x.mm(W_target) + b_target.item()


def poly_desc(W, b):
    """Creates a string description of a polynomial."""
    # 创建描述多项式的字符串。
    result = 'y = '
    for i, w in enumerate(W):
        result += '{:+.2f} x^{} '.format(w, i + 1)
    result += '{:+.2f}'.format(b[0])
    return result


def get_batch(batch_size=32):
    """Builds a batch i.e. (x, f(x)) pair."""
    # 构建一个批次，即一组 (x, f(x)) 数据对。
    random = torch.randn(batch_size)
    x = make_features(random)
    y = f(x)
    return x, y


# Define model
# 定义模型。
fc = torch.nn.Linear(W_target.size(0), 1)

for batch_idx in count(1):
    # Get data
    # 获取数据。
    batch_x, batch_y = get_batch()

    # Reset gradients
    # 清空梯度。
    fc.zero_grad()

    # Forward pass
    # 前向传播。
    output = F.smooth_l1_loss(fc(batch_x), batch_y)
    loss = output.item()

    # Backward pass
    # 反向传播。
    output.backward()

    # Apply gradients
    # 应用梯度更新。
    for param in fc.parameters():
        param.data.add_(-0.1 * param.grad)

    # Stop criterion
    # 停止条件。
    if loss < 1e-3:
        break

print('Loss: {:.6f} after {} batches'.format(loss, batch_idx))
print('==> Learned function:\t' + poly_desc(fc.weight.view(-1), fc.bias))
print('==> Actual function:\t' + poly_desc(W_target.view(-1), b_target))
