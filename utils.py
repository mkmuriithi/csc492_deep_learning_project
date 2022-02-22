import yfinance as yf
import pandas as pd
import numpy as np


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


def get_master_dataset(tickers_list=[]):
    """
        In order to iterate through the master dataset using tickers in the dataset, please do df.group_by(level=0)
        """

    if tickers_list.isempty():
        tickers_list = get_list_of_tickers()

    master_dataset = yf.download(
        tickers=tickers_list,
        period="15y",
        interval='1d',
        group_by='ticker',
        auto_adjust=True,  # adjusts for splits and dividends, necessary for accuracy
        threads=True,
    )
    return master_dataset
