# author: roczhang
# file: dataset_load.py
# time: 2021/04/16
import os
import torch
import scipy.io as scio
import scipy
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms

transform = transforms.Compose(
    [transforms.ToTensor()])

class CustomDataset(Dataset):
    def __init__(self, train_dir, labels_dir):
        labels_dict = scio.loadmat(labels_dir)
        labels_key = list(labels_dict)[-1]
        self.labels = labels_dict[labels_key]

        self.train_dir = train_dir

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        train_dict = scio.loadmat(self.train_dir)
        train_key = list(train_dict)[-1]
        train_data = train_dict[train_key]
        labels = self.labels
        samples = {"train": train_data, "labels": labels}
        return samples


train_data = CustomDataset("/data/file/classification_data/SJ15/trainData.mat", "/data/file/classification_data/SJ15/trainlabel.mat",
                           transforms=transforms)
test_data = CustomDataset("/data/file/classification_data/SJ15/testData.mat", "/data/file/classification_data/SJ15/testlabel.mat",
                          transforms=transforms)
train_dataloader = DataLoader(train_data, batch_size=32)
test_dataloader = DataLoader(test_data, batch_size=32)


train_features, train_labels = next(iter(train_dataloader)).values()
print(train_features.shape)
print(train_labels.shape)

# 定义网络
import torch
import torch.nn as nn


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(194, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 20)
        self.fc4 = nn.Linear(25, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.fc4(x)
        return x

net = Net()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

running_loss = 0.0
for epoch in range(1):
    for i, data in enumerate(train_dataloader):
        inputs, labels = data.values()
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 1000 == 999:
            print("training loss", running_loss / 1000, epoch * len(train_dataloader) + i)
            # print("acctuals",)
        running_loss = 0.0

print("Finished Training")