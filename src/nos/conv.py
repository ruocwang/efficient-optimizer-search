import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.insert(0, '../')


class ConvDefault(nn.Module):

    def __init__(self):
        super(ConvDefault, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        # self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        # self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.bn2   = nn.BatchNorm2d(64)
        
        self.fc1 = nn.Linear(65536, 10)
        # self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.bn2(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x


class ConvQuoc(nn.Module):

    def __init__(self):
        super(ConvQuoc, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.bn2   = nn.BatchNorm2d(32)
        
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        ## TODO under dev ## quoc: not sure if they have pool
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.bn2(x)
        x = self.pool2(x)

        x = torch.flatten(x, 1)
        
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


class Conv(nn.Module):

    def __init__(self):
        super(Conv, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.bn2   = nn.BatchNorm2d(32)
        
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        ## TODO under dev ## quoc: not sure if they have pool
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = self.bn2(x)

        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


if __name__ == '__main__':
    net = Conv()
    x = torch.ones((128, 3, 32, 32))
    out = net(x)
    print(out.shape)
