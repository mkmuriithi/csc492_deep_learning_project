import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from ta.momentum import RSIIndicator as rsi
from ta.momentum import AwesomeOscillatorIndicator
from ta.trend import sma_indicator as sma
from ta.trend import EMAIndicator as ema
from ta.volume import VolumeWeightedAveragePrice as vwap
from StockDataLoader import *
import os
from tqdm import tqdm

def add_features(data):
    data["Sma_10"] = sma(data.Close, window=10, fillna=False)
    # data["Sma_20"] = sma(data.Close, window=20, fillna=False)
    # data["Sma_50"] = sma(data.Close, window=50, fillna=False)
    # data["Sma_100"] = sma(data.Close, window=100, fillna=False)
    # data["Sma_200"] = sma(data.Close, window=200, fillna=False)
    data["Ema_10"] = ema(data.Close, window=10, fillna=False).ema_indicator()
    # data["Ema_20"] = ema(data.Close, window=20, fillna=False).ema_indicator()
    # data["Ema_50"] = ema(data.Close, window=50, fillna=False).ema_indicator()
    # data["Ema_100"] = ema(data.Close, window=100, fillna=False).ema_indicator()
    # data["Ema_200"] = ema(data.Close, window=200, fillna=False).ema_indicator()
    data["RSI"] = rsi(data.Close, window=6, fillna=False).rsi()
    data["VWAP"] = vwap(high=data.High, low=data.Low,
                        close=data.Close, volume=data.Volume, window=14, fillna=False).volume_weighted_average_price()

    return (data)


def get_treated(data, to_daily_returns=True, features_to_exclude=['Date']):
    # get percentage change first
    for feature in data.columns:
        if feature not in features_to_exclude:
            data[feature] = data[feature].pct_change()
    data.dropna(inplace=True)
    return data


def get_dataset(single=True, subset=False, ticker_to_train="XOM"):
    '''
    If single stock, then only do apple, and second parameter is not relevant
    If mulitstock, then second parameter determines if a subset of the stocks will be selected to be trained on
    on or if all stocks in the nyse_list will be trained on
    NOTE that if 'single' is False, ticker_to_train is ignored
    '''
    data = None
    if single:
        data = pd.read_csv(f'./Data/{ticker_to_train}.csv')
        data["Date"] = pd.to_datetime(data['Date'])
        return data
    else:
        return get_multistock_dict(subset)


def get_multistock_dict(subset):
    # read stock_list
    temp = os.listdir('./Data')
    stock_dict = {}
    list_of_stocks_included = []
    # randomly choose stocks
    n_years_ago_date = get_date_n_years_ago(3)
    if subset:
        pass
    else:
        # add all stocks to dict
        for stock_dir_name in tqdm(temp):
            stock_name = stock_dir_name.split(".")[0]
            df = pd.read_csv(f'./Data/{stock_dir_name}')
            df["Date"] = pd.to_datetime(df['Date'])
            if df['Date'][0].date() > n_years_ago_date: #only if we have at least n years worth of data
                continue
            else:
                stock_dict[stock_name] = df
                list_of_stocks_included.append(stock_name)

    return stock_dict

def get_date_n_years_ago(n):

    import datetime
    today = datetime.datetime.today().date()
    days_per_year = 365.24
    years_ago = n
    n_years_ago = today - datetime.timedelta(days=(days_per_year * years_ago))
    return n_years_ago

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
