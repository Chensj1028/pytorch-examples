"""
This python script converts the network into Script Module
"""
# 此 Python 脚本将网络转换为 ScriptModule。
import torch
from torchvision import models

# Download and load the pre-trained model
# 下载并加载预训练模型。
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

# Set upgrading the gradients to False
# 关闭梯度更新。
for param in model.parameters():
	param.requires_grad = False

# Save the model except the final FC Layer
# 保存除最后一个全连接层以外的模型。
resnet18 = torch.nn.Sequential(*list(model.children())[:-1])

example_input = torch.rand(1, 3, 224, 224)
script_module = torch.jit.trace(resnet18, example_input)
script_module.save('resnet18_without_last_layer.pt')
