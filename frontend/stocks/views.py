from django.shortcuts import render
from django.http import HttpResponseRedirect
from .forms import TickerForm
from .yfinance import get_30_days_data, get_ticker_info
 
def index(request):
    if request.method == 'POST':
        form = TickerForm(request.POST)
        if form.is_valid():
                ticker = request.POST['ticker'] # extract user input
                return HttpResponseRedirect(ticker)
    else:
        form = TickerForm()
    return render(request, 'index.html', {'form': form})

def ticker(request, ticker_id):
    context = {}
    context['ticker'] = ticker_id
    context['ticker_30_days_data_html'] = get_30_days_data(ticker_id).to_html()
    # todo: prediction. plotly
    context['ticker_info'] = get_ticker_info(ticker_id)
    return render(request, 'ticker.html', context)