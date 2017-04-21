from torch.autograd import Function
from torch import nn
import torch
import numpy as np


def compute_euclidean_distance(x, y):
    return torch.sqrt(torch.sum((x - y)**2))


class ContrastiveLossF(Function):
    def __init__(self, margin=1):
        super(ContrastiveLossF, self).__init__()
        self.margin = margin

    def forward(self, input1, input2, label):
        d = compute_euclidean_distance(input1, input2)

        between_class = label * d ** 2  # (1-Y)*(d^2) ???
        max_part = (max(self.margin - d, 0)) ** 2

        within_class = (1 - label) * max_part  # (Y) * max((margin - d)^2, 0)

        output = 0.5 * torch.mean(within_class + between_class)
        self.save_for_backward(input1, input2, label, d)

        return input1.new((output,))

    def backward(self, grad_output):
        input1, input2, y, d = self.saved_tensors
        grad_input1 = input1.new().resize_as_(input1)
        grad_input2 = input2.new().resize_as_(input2)

        dist = input1.clone()
        dist.add_(-1, input2)
        dist.mul_(-1).mul_(y)
        dist.add_(self.margin)
        mask = dist.ge(0)

        grad_input1.copy_(mask)
        grad_input1.mul_(-1).mul_(y)
        grad_input2.copy_(mask)
        grad_input2.mul_(y)

        if self.size_average:
            grad_input1.div_(y.size(0))
            grad_input2.div_(y.size(0))

        return grad_input1, grad_input2, None


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, input1, input2, target):
        return ContrastiveLossF(self.margin)(input1, input2, target)