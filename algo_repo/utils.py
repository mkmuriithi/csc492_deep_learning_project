import yfinance as yf
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt


def get_list_of_tickers():
    nyse_stocks_tickers = []
    nasdaq_stocks_tickers = []

    with open("nyse_stocks_list.txt", 'r') as nyse_file, open("nasdaq_stock_list.txt", 'r') as nasdaq_file:
        nyse_stocks_tickers = nyse_file.read().split("\n")
        nasdaq_stocks_tickers = nasdaq_file.read().split("\n")

    '''
    Step 2: Ticker string cleaning:
    # 1) Remove Securities. If tickers have a caret (^) then they are indices, not securities, meaning we cannot directly invest in it
    # 2) Nasdaq's convention (where we got the .txt list uses '/' wheras yfinance uses '-'. Eg AKO/A in .txt -> AKO-A in yfinance
    '''
    nyse_stocks_list = []
    nasdaq_stocks_list = []

    '''
    There are a couple types of tickers that we will not consider:
    Remove tickers with 

    '''
    for ticker in nyse_stocks_tickers:
        if '^' not in ticker:
            if '/' in ticker:
                ticker = ticker.replace('/', '-')
            nyse_stocks_list.append(ticker)

    for ticker in nasdaq_stocks_tickers:
        if '^' not in ticker:
            if '/' in ticker:
                ticker = ticker.replace('/', '-')
            nasdaq_stocks_list.append(ticker)

    tickers_list = nyse_stocks_list
    tickers_list.extend(nasdaq_stocks_list)
    return tickers_list


#yfinance will act up and misbehave when we try to download, so try to circumvent this
def get_master_dataset(tickers_list=[], timestart="2007-01-01", timeend="2021-12-31"):
    """
        In order to iterate through the master datasetusing tickers in the dataset, please do df.group_by(level=0)
        """

    if not tickers_list:
        tickers_list = get_list_of_tickers()

    master_dataset = yf.download(
        tickers=tickers_list,
        start=timestart,
        end=timeend,
        interval='1d',
        group_by='ticker',
        auto_adjust=True,  # adjusts for splits and dividends, necessary for accuracy
        threads=True,
    )
    return master_dataset


def get_historic_data(symbol):
    ticker = symbol
    iex_api_key = 'Tsk_30a2677082d54c7b8697675d84baf94b'
    api_url = f'https://sandbox.iexapis.com/stable/stock/{ticker}/chart/max?token={iex_api_key}'
    df = requests.get(api_url).json()

    date = []
    open = []
    high = []
    low = []
    close = []

    for i in range(len(df)):
        date.append(df[i]['date'])
        open.append(df[i]['open'])
        high.append(df[i]['high'])
        low.append(df[i]['low'])
        close.append(df[i]['close'])

    date_df = pd.DataFrame(date).rename(columns={0: 'date'})
    open_df = pd.DataFrame(open).rename(columns={0: 'open'})
    open_df = pd.DataFrame(open).rename(columns={0: 'open'})
    high_df = pd.DataFrame(high).rename(columns={0: 'high'})
    low_df = pd.DataFrame(low).rename(columns={0: 'low'})
    close_df = pd.DataFrame(close).rename(columns={0: 'close'})

    frames = [date_df, open_df, high_df, low_df, close_df]
    df = pd.concat(frames, axis=1, join='inner')
    df = df.set_index('date')

    df['open'].plot()
    plt.title('{} Historical Prices'.format(ticker), fontsize=18)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Stock Price', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()

    return df