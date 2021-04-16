# author:roczhang
# file:BP_cls.py
# time:2021/04/16
import numpy as np
import torch
from torchvision.datasets import mnist  # 导入 pytorch 内置的 mnist 数据

from torch import nn
from torch.autograd import Variable

from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

class BPNNModel(torch.nn.Module):
    def __init__(self):
        # 调用父类的初始化函数，必须要的
        super(BPNNModel, self).__init__()

        # 创建四个Sequential对象，Sequential是一个时序容器，将里面的小的模型按照序列建立网络
        self.layer1 = nn.Sequential(nn.Linear(784, 400), nn.ReLU())
        self.layer2 = nn.Sequential(nn.Linear(400, 200), nn.ReLU())
        self.layer3 = nn.Sequential(nn.Linear(200, 100), nn.ReLU())
        self.layer4 = nn.Sequential(nn.Linear(100, 10))

    def forward(self, img):
        # 每一个时序容器都是callable的，因此用法也是一样。
        img = self.layer1(img)
        img = self.layer2(img)
        img = self.layer3(img)
        img = self.layer4(img)
        return img

