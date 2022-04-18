from this import d
import yfinance as yf
import sys
import torch
from pathlib import Path


from sklearn.preprocessing import MinMaxScaler
sys.path.append("/home/kagema/Documents/CSC 492/csc492_deep_learning_project/algo_repo")

from train import *
from train_multiple import * 
from data_stuff import *



class Data:
    
    #pass ticker when initializing
    def __init__(self, ticker, n_days=45):
        self.ticker = ticker
        self.info = yf.Ticker(ticker).info
        self.data = yf.download(ticker, period=f"{n_days}d", auto_adjust=True)
    
    def get_n_days_data(self, n_days=30):
        return self.data.iloc[-n_days:]

    def get_transformed_data(self):
        df = self.data.reset_index()
        df = treat_single_stock(df) #returns stock changed to percentage change
        X_data = df.drop(columns=["Target"])
        y_data = df[["Target"]]
    
    
    #minmax scaling target, return to reverse scaling
        data_tensor = torch.Tensor(X_data.values)
        mask = torch.zeros(data_tensor.shape[0], data_tensor.shape[0])
        data_tensor = data_tensor.unsqueeze(0)
        return data_tensor, mask

    def get_ticker_info(self):
        return self.info
    

