from django.shortcuts import render
from django.http import HttpResponseRedirect
from .forms import TickerForm
from .yfinance import get_30_days_data, get_ticker_info
import logging
# TODO: import model
 
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
    dataframe_30_days_data = get_30_days_data(ticker_id) #dataframe
    raw_30_days_data =  dataframe_30_days_data.to_numpy()
    raw_30_days_adj_close = raw_30_days_data[:, 4]
    # TODO: pass to model
    ## load model.... for example
        # model = TheModelClass(*args, **kwargs)
        # model.load_state_dict(torch.load(PATH))
        # model.eval()

    # prediction = model.predict(raw_30_days_adj_close)
    # context['prediction'] = prediction
    context['raw_30_days_adjclose'] = raw_30_days_data[:, 4]
    context['ticker_30_days_data_html'] = dataframe_30_days_data.to_html()
    # todo: prediction. plotly
    context['ticker_info'] = get_ticker_info(ticker_id)
    return render(request, 'ticker.html', context)