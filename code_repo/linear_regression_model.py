import os, torch
from torch import nn
from torch.utils.data import DataLoader


class LinearRegression(nn.Module):
    def __init__(self, in_features, out_features):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        t = self.linear(x)
        return t
