import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from ta.momentum import RSIIndicator as rsi
from ta.momentum import AwesomeOscillatorIndicator
from ta.trend import sma_indicator as sma
from ta.trend import EMAIndicator as ema
from ta.volume import VolumeWeightedAveragePrice as vwap


def add_features(data):
    data["Sma_10"] = sma(data.Close, window=10, fillna=False)
    #data["Sma_20"] = sma(data.Close, window=20, fillna=False)
    #data["Sma_50"] = sma(data.Close, window=50, fillna=False)
    #data["Sma_100"] = sma(data.Close, window=100, fillna=False)
    #data["Sma_200"] = sma(data.Close, window=200, fillna=False)
    data["Ema_10"] = ema(data.Close, window=10, fillna=False).ema_indicator()
    #data["Ema_20"] = ema(data.Close, window=20, fillna=False).ema_indicator()
    #data["Ema_50"] = ema(data.Close, window=50, fillna=False).ema_indicator()
    #data["Ema_100"] = ema(data.Close, window=100, fillna=False).ema_indicator()
    #data["Ema_200"] = ema(data.Close, window=200, fillna=False).ema_indicator()
    data["RSI"] = rsi(data.Close, window=6, fillna=False).rsi()
    data["VWAP"] = vwap(high=data.High, low=data.Low,
                        close=data.Close, volume=data.Volume, window=14, fillna=False).volume_weighted_average_price()

    return(data)


def get_treated(data, to_daily_returns=True, features_to_exclude = ['Date']):
    # get percentage change first
    for feature in data.columns:
        if feature not in features_to_exclude:
            data[feature] = data[feature].pct_change()
    data.dropna(inplace=True)
    return data

