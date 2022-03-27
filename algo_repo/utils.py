import yfinance as yf
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from StockDataLoader import StockDataset
from sklearn.preprocessing import MinMaxScaler


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


# yfinance will act up and misbehave when we try to download, so try to circumvent this
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
    params = {'token': 'pk_ea807dc493764e34917c4d18922a874a'}
    sandbox_param = {"token": 'Tsk_30a2677082d54c7b8697675d84baf94b'}
    # sandapi_url = f'https://sandbox.iexapis.com/stable/stock/{ticker}/chart/max?token={iex_api_key}'
    api_url = f'https://sandbox.iexapis.com/stable/stock/{ticker}/chart/max'
    df = requests.get(api_url, params=sandbox_param).json()
    print("new shit is working")

    date = []
    open = []  # split adjusted prices
    high = []  # split adjusted prices
    low = []  # split adjusted prices
    close = []  # split adjsuted prices
    volume = []  # spit adjusted volume
    change = []  # change from previous trading day
    changePercent = []  # change percent from previous trading day
    uOpen = []  # unadjusted price
    uClose = []  # unadjusted price
    uHigh = []  # unadjusted price
    uLow = []  # unadjusted price
    uVolume = []  # unadjusted volume
    fullOpen = []  # fully adjusted price
    fullClose = []  # fully adjusted price
    fullHigh = []  # fully adjusted price
    fullLow = []  # fully adjusted price
    fullVolume = []  # fully adjusted volume

    for i in range(len(df)):
        date.append(df[i]['date'])
        open.append(df[i]['open'])
        high.append(df[i]['high'])
        low.append(df[i]['low'])
        close.append(df[i]['close'])
        volume.append(df[i]["volume"])
        change.append(df[i]["change"])
        changePercent.append(df[i]["changePercent"])
        uOpen.append(df[i]['uOpen'])
        uClose.append(df[i]['uClose'])
        uHigh.append(df[i]['uHigh'])
        uLow.append(df[i]['uLow'])
        uVolume.append(df[i]['uVolume'])
        fullOpen.append(df[i]['fOpen'])
        fullClose.append(df[i]['fClose'])
        fullHigh.append(df[i]['fHigh'])
        fullLow.append(df[i]['fLow'])
        fullVolume.append(df[i]["fVolume"])
    date_df = pd.DataFrame(date).rename(columns={0: 'date'})
    open_df = pd.DataFrame(open).rename(columns={0: 'open'})
    high_df = pd.DataFrame(high).rename(columns={0: 'high'})
    low_df = pd.DataFrame(low).rename(columns={0: 'low'})
    close_df = pd.DataFrame(close).rename(columns={0: 'close'})
    volume_df = pd.DataFrame(volume).rename(columns={0: 'volume'})
    change_df = pd.DataFrame(change).rename(columns={0: 'change'})
    changePercent_df = pd.DataFrame(changePercent).rename(columns={0: 'changePercent'})
    uOpen_df = pd.DataFrame(uOpen).rename(columns={0: 'uOpen'})
    uClose_df = pd.DataFrame(uClose).rename(columns={0: "uClose"})
    uHigh_df = pd.DataFrame(uHigh).rename(columns={0: "uHigh"})
    uLow_df = pd.DataFrame(uLow).rename(columns={0: "uLow"})
    uVolume_df = pd.DataFrame(uVolume).rename(columns={0: "uVolume"})
    fullOpen_df = pd.DataFrame(fullOpen).rename(columns={0: "fullOpen"})
    fullClose_df = pd.DataFrame(fullClose).rename(columns={0: "fullClose"})
    fullHigh_df = pd.DataFrame(fullHigh).rename(columns={0: "fullHigh"})
    fullLow_df = pd.DataFrame(fullLow).rename(columns={0: "fullLow"})
    fullVolume_df = pd.DataFrame(fullVolume).rename(columns={0: "fullVolume"})

    frames = [date_df, open_df, high_df, low_df, close_df, volume_df, change_df, changePercent_df, uOpen_df,
              uClose_df, uHigh_df, uLow_df, uVolume_df, fullOpen_df, fullClose_df, fullHigh_df, fullLow_df,
              fullVolume_df]

    df = pd.concat(frames, axis=1, join='inner')
    df = df.set_index('date')
    '''
    df['open'].plot()
    plt.title('{} Historical Prices'.format(ticker), fontsize=18)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Stock Price', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()
    '''
    print("good")
    return df


def split_data(data, window=30, minmax=True):
    data.dropna(inplace=True)
    X = data.drop(["Target"], axis=1)
    y = data[["Target"]]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, shuffle=False)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42, shuffle=False)

    if minmax:
        columns = X_train.columns
        index = X_train.index
        X_train = pd.DataFrame(MinMaxScaler().fit_transform(X_train), columns=columns, index=index)

        columns = X_val.columns
        index = X_val.index
        X_val = pd.DataFrame(MinMaxScaler().fit_transform(X_val), columns=columns, index=index)

        columns = X_test.columns
        index = X_test.index
        X_test = pd.DataFrame(MinMaxScaler().fit_transform(X_test), columns=columns, index=index)

        columns = y_train.columns
        index = y_train.index
        y_train = pd.DataFrame(MinMaxScaler().fit_transform(y_train), columns=columns, index=index)

        columns = y_val.columns
        index = y_val.index
        y_val = pd.DataFrame(MinMaxScaler().fit_transform(y_val), columns=columns, index=index)

        columns = y_test.columns
        index = y_test.index
        y_test = pd.DataFrame(MinMaxScaler().fit_transform(y_test), columns=columns, index=index)

    dataset_train = StockDataset(X_train, y_train, window)
    dataset_val = StockDataset(X_val, y_val, window)
    dataset_test = StockDataset(X_test, y_test, window)

    return dataset_train, dataset_val, dataset_test
