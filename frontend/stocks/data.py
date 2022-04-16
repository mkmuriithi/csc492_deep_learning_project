import yfinance as yf
import sys
import torch

sys.path.append("/home/kagema/Documents/CSC 492/csc492_deep_learning_project/algo_repo")
from train import *
from train_multiple import *
from data_stuff import *


def get_ticker_info(ticker):
    '''
    returns dictionary of info, 
    we are interested in keys 'shortName', 'longBusinessSummary'
    '''
    # todo: check valid ticker string
    ticker = yf.Ticker(ticker)
    return ticker.info

def get_n_days_data(ticker, n_days=30):
    '''
    returns a dataframe of time day data
    Date | Open | High | Low | Close | Volume
    Note that data is OHLC is automatically adjusted
    '''
    # todo: check valid ticker string
    data = yf.download(ticker, period=f"{n_days}d", auto_adjust=True)
    return data  # dataframe


def get_transformed_data(data):
    '''
    Accepts pandas dataframe that is direct output of yfinance api for a single stock
    NOTE: in order to use the most recent 30 days of data, the stock chosen needs to have at least
    44 days worth of data so that the features necessary for predictive power can be engineered

    Returns data that can be passed to the model
    '''

    data.reset_index(inplace=True)
    data = treat_single_stock(data)
    data.drop(columns=["Target"], inplace=True)
    print(data.head())
    data_tensor = torch.Tensor(data.values)
    mask = torch.zeros(data_tensor.shape[0], data_tensor.shape[0])

    data_tensor = data_tensor.unsqueeze(0)
    
    
    return data_tensor, mask

