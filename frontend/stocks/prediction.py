import sys
from stocks.data import get_transformed_data
sys.path.append("/home/kagema/Documents/CSC 492/csc492_deep_learning_project/algo_repo")

from train import *
from train_multiple import *
from data_stuff import *
import torch
from .data import get_n_days_data
from .data import get_transformed_data



single_path = "/home/kagema/Documents/CSC 492/csc492_deep_learning_project/algo_repo/model_pickles/model_date_04_14_2022_time_19_51.pt"

def get_prediction(path_of_pickle, ticker):
    '''
    Passing in relative path
    '''

    #get the data
    data = get_n_days_data(ticker, 44) #needs 44 days of data for feature engineering

    X, mask = get_transformed_data(data)
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
    return  out

#def reverse_normalization()