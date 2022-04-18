import sys
from pathlib import Path

import torch

from .data import Data

from train import *
from train_multiple import *
from data_stuff import *

def get_prediction(path_of_pickle, df):
    '''
    Passing in relative path
    df is the Data class we defined
    '''

    #get the data
    data = df.get_n_days_data(44)
    #data = get_n_days_data(ticker, 44) #needs 44 days of data for feature engineering

    X, mask = df.get_transformed_data()
    #make the model
    model = TransformerModel(transf_params)
    model.load_state_dict(torch.load(path_of_pickle))
    model.eval()

    if torch.cuda.is_available():
        X = X.cuda()
        mask = mask.cuda()
        model = model.cuda()
    
    out = model(X, mask)
    out = out.cpu()
    out = out.squeeze()
    out = out.detach()
    out = out.numpy()

    percentage_out = out*100
    absolute_out = out * float(df.get_n_days_data(1).Close)
    return  percentage_out, absolute_out

#def reverse_normalization()