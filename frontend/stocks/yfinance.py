import yfinance as yf

def get_ticker_info(ticker):
    '''
    returns dictionary of info, 
    we are interested in keys 'shortName', 'longBusinessSummary'
    '''
    # todo: check valid ticker string
    ticker = yf.Ticker(ticker)
    return ticker.info

def get_30_days_data(ticker):
    '''
    returns a dataframe of 30 day data
    Date | Open | High | Low | Close | Adj Close | Volume
    '''
    # todo: check valid ticker string
    # todo. only return adj close.
    data = yf.download(ticker, period="30d") # dataframe
    return data 
