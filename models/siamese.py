import torch
import torchvision
from torch import nn
import copy
import torch.nn.functional as F


class Siamese(nn.Module):
    def __init__(self):
        super(Siamese, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.MaxPool2d(2, stride=2))

        self.fc1 = nn.Sequential(
            nn.Linear(50 * 4 * 4, 500),
            nn.ReLU(inplace=True),
            nn.Linear(500, 10),
            nn.Linear(10, 2))

        # self.cnn2 = self.cnn1
        # self.fc2= self.fc1

        # print self.cnn1 is self.cnn2

    def forward(self, input1, input2):
        input1 = self.cnn1(input1)
        input1 = input1.view(input1.size()[0], -1)
        input1 = self.fc1(input1)

        input2 = self.cnn1(input2)
        input2 = input2.view(input2.size()[0], -1)
        input2 = self.fc1(input2)

        output = torch.sqrt(torch.sum((input1 - input2) * (input1 - input2), 1))
        return output


