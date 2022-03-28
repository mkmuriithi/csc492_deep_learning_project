import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def get_treated(data, to_daily_returns=True):
    # get percentage change first
    temp_holder = data['Date']
    if to_daily_returns:
        data = _get_daily_returns(data)
    data['Date'] = temp_holder
    data.dropna(inplace=True)
    return data


def _get_daily_returns(data):
    # use pct change for relevant columns
    features = ['Open', 'Low', 'High', 'Close', 'Volume']  # in case we add more columns we list relevant ones here
    for feature in features:
        data[feature] = data[feature].pct_change()
    return data
