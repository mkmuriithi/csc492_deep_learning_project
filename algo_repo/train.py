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
import matplotlib.pyplot as plt

from StockDataLoader import StockDataset
from torch.utils.data import DataLoader
import torch.optim as optim
from sklearn.model_selection import train_test_split


import transformer
import importlib
importlib.reload(transformer)
class TransformerModel(nn.Module):
    def __init__(self, params):
        super(TransformerModel, self).__init__()
        self.transf = transformer.TransformerModel(n_layers=params.n_layers,
                                                   num_heads=params.num_heads,
                                                   model_dim=params.model_dim,
                                                   forward_dim=params.forward_dim,
                                                   output_dim=16,
                                                   dropout=params.dropout)
        self.linear = nn.Linear(16, params.output_dim)
    def forward(self, x):
        transf_out = self.transf(x)
        out = self.linear(transf_out)
        return out


def train(model, data, optimizer='adam', batch_size=16, learning_rate=0.1, momentum=0.9, num_epochs=10, weight_decay=0.0):

    #create training, valid and test sets of StockDataset type data
    train_custom, valid_custom, test_custom= split_data(data)

    #create loaders
    train_dataloader = DataLoader(train_custom, batch_size=16, shuffle=False) #returns the X and associated y prediction
    val_dataloader = DataLoader(valid_custom, batch_size=16, shuffle=False) #does same
    valid_iterator = iter(val_dataloader)
    optimizer = optim.Adam(model.parameters(),
                          lr = learning_rate,
                           weight_decay = weight_decay)

    #track learning curve
    criterion = nn.MSELoss(reduction="mean")
    iters, train_losses, val_losses = [], [], []
    #train
    n = 0
    for epoch in range(0, num_epochs):
        print(f'Epoch {epoch} training beginning...')
        for X,y in iter(train_dataloader):
            if len(X) < batch_size:
                continue
            model.train() #annotate for train
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            iters.append(n)
            train_losses.append(float(loss)) #average loss
            print(f'iter{n}')
            #predict validation
            for X_val, y_val in iter(val_dataloader):

                model.eval() #annotate for test
                val_out = model(X_val)
                val_loss = criterion(val_out, y_val)
                val_losses.append(val_loss)

            #save steo
    print(f'Final Training Loss: {train_losses[-1]}')
    print(f'Final Val Accuracy {val_losses}')
    #graph loss
    plt.title("Learning Loss")
    plt.plot(iters, train_losses, label='Train')
    plt.plot(iters, val_losses, label='Validation')
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.show()

class transf_params:
    n_layers = 4
    num_heads = 6
    model_dim = 6  # nr of features
    forward_dim = 2048
    output_dim = 1
    dropout = 0
    n_epochs = 10
    lr = 0.01

def split_data(data):


    X = data.drop(["Target"], axis=1)
    y = data["Target"]
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.1, random_state=42, shuffle=False)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42, shuffle=False)

    dataset_train = StockDataset(X_train, y_train, 30)
    dataset_val = StockDataset(X_val, y_val, 30)
    dataset_test = StockDataset(X_test, y_test, 30)

    return dataset_train, dataset_val, dataset_test

if __name__ == '__main__':

    import yfinance as yf
    data = yf.download(tickers="AAPL", period='max', interval='1d', groupby='ticker', auto_adjust='True')
    data.reset_index(inplace=True)
    data.index = data.index.set_names(["order"])
    data.reset_index(inplace=True)  # to keep up with order
    data['Target'] = data["Close"].shift(-1)
    data["Date"] = data["Date"].apply(lambda x: x.value / 10 ** 9)
    data.drop(columns=['order'], inplace=True)


    X = data.drop(["Target"], axis=1)
    y = data["Target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, shuffle=False)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42, shuffle=False)



    dataset_train = StockDataset(X_train, y_train, 30)
    dataset_val = StockDataset(X_val, y_val, 30)

    train_dataloader = DataLoader(dataset_train, batch_size=16, shuffle=False)
    val_dataloader = DataLoader(dataset_val, batch_size=16, shuffle=False)

    model = TransformerModel(transf_params)
    train(model, data)