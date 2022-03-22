import pandas as pd
import numpy as np
import matplotlib.pyplot
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data.encoders import TorchNormalizer
import torch.nn as nn
import torch.optim as optim
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from pytorch_forecasting.models.base_model import BaseModel, BaseModelWithCovariates


class StockDataset(Dataset):
    """
    A dataset that extends the Pytorch Dataset class, represents a dataset of a single stock

    '''
    Attributes
    ----------
    x: Pandas DataFrame
        features
    y: Pandas DataFrame
        target
    sequence_length: int
        how many days used to predict subsequent days (window length)

    Methods
    --------


    Examples
    --------
    dataset_train = StockDataset(X_train, y_train, 30)
    dataset_val = StockDataset(X_val, y_val, 30)

    train_dataloader = DataLoader(dataset_train, batch_size=16, shuffle=True)
    val_dataloader = DataLoader(dataset_val, batch_size=16, shuffle=True)

    train_feeder = iter(train_loader)
    next(train_feeder) #returns batch of sequences, 30 day sequences in this example
    """

    # must overwrite __getitem__, __len__, __
    def __init__(self, x, y, sequence_length):
        self.x = x
        self.y = y  # shifted close price
        self.seq_length = sequence_length  # can be thought of as the window size

    def __len__(self):
        # length means how many sequences
        return self.x.shape[0] - self.seq_length

    # assuming __getsize__ will be used by data loader to figure out when to stop

    def __getitem__(self, idx):
        # return sequences
        # return
        # ave to return the y corresponding to the last element in the sequence NOT first element
        return Tensor(self.x.iloc[[idx]].values), Tensor(self.y.iloc[[idx + self.seq_length]].values)

