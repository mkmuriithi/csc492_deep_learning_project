from django.shortcuts import render
from django.http import HttpResponseRedirect
from .forms import TickerForm
#from .data import get_n_days_data, get_ticker_info
from .data import Data
import logging
from .prediction import *

import os
from .algo_files.train import *
from .algo_files.train_multiple import * 
from .algo_files.data_stuff import *


# TODO: import model
 
 #Dango views receve wen request and return web response
 #Here we return a html page, where the context key-value pairs we add define
 # what the passed over html will return
def index(request):
    if request.method == 'POST':
        form = TickerForm(request.POST)
        if form.is_valid():
                ticker = request.POST['ticker'] # extract user input
                return HttpResponseRedirect(ticker)
    else:
        form = TickerForm()
    return render(request, 'index.html', {'form': form})

# TODO:
def ticker(request, ticker_id):
    context = {}
    context['ticker'] = ticker_id

    #create class
    df = Data(ticker_id)
    dataframe_30_days_data = df.get_n_days_data(30)
    #dataframe_30_days_data = get_n_days_data(ticker_id) #dataframe
    raw_30_days_data =  dataframe_30_days_data.to_numpy() #already adjusted
    raw_30_days_adj_close = raw_30_days_data
    #raw_30_days_adj_close = raw_30_days_data[:, 4]
    # TODO: pass to model
    #For a given stock, we will have individually trained models for that stock, and use the ticker ID and do
    # string formatting to get the appropriate single model
    single_stock_path = f'./final_models/{ticker_id}.pt'
    multi_stock_path = f'./final_models/general_model.pt'


    single_percentage_out, single_absolute_out = get_prediction(single_stock_path, df)
    multiple_precentage_out, multiple_absolute_out = get_prediction(multi_stock_path, df)
    #prediction_single = get_prediction(single_path, ticker_id)
    #prediction_multiple = get_prediction(multi_stock_path, ticker_id)


    context['single_pred_percent'] =  single_percentage_out
    context['single_pred_abs'] = single_absolute_out
    context['multiple_pred_percent'] = multiple_precentage_out
    context['multiple_pred_abs'] = multiple_absolute_out
    context["previous_price"] = round(float(df.get_n_days_data(1).Close), 2)

    #make prediction

        # model = TheModelClass(*args, **kwargs)
        # model.load_state_dict(torch.load(PATH))
        # model.eval()git 

    # prediction = model.predict(raw_30_days_adj_close)
    # context['prediction'] = prediction
    context['raw_30_days_adjclose'] = raw_30_days_data[:, 4]
    context['ticker_30_days_data_html'] = dataframe_30_days_data.to_html()
    # todo: prediction. plotly
    context['ticker_info'] = df.get_ticker_info()
    return render(request, 'ticker.html', context)







