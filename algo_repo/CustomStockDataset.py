import numpy as np
import torch
from torch.utils.data import Dataset

import os
from torch import nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
pd.set_option('display.max_columns', None)

class CustomStockDataset(Dataset):

    def __init__(self, dataset, forecast_window=1, path=''):
        self.path = path
        self.forecast_window = forecast_window
        self.dataset = dataset #pandas multilevel dataframe

    def __len__(self):
        return(len(self.dataset))

    def __getitem__(self, stock=[]):
        #returns dataset of requested stock as torch tensor
        requested_datasets = {}
        if not stock:
            return {}
        else:
            for stock in dat