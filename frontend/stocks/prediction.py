import sys
from pathlib import Path

import torch

from .data import Data

from .algo_files.train import *
from .algo_files.train_multiple import * 
from .algo_files.data_stuff import *

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
    
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(path_of_pickle))
    else: #if only cpu
        model.load_state_dict(torch.load(path_of_pickle, map_location=torch.device('cpu')))

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

    last_price = float(df.get_n_days_data(1).Close)

    percentage_out = round(out*100,2)

    absolute_out = last_price * (1 + out)
    absolute_out = round(absolute_out, 2)


    return  percentage_out, absolute_out

#def reverse_normalization()