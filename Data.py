import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import seaborn
from utils import *


# note that tickers


class Data:

    def __init__(self, data_path=None, tickers_list=None):
        if data_path is None:
            if tickers_list is None:
                self.data = get_master_dataset()
            else:
                self.data = get_master_dataset(tickers_list)
    def treat_missing(self, method):
        #fill missing values with method
        #for now only support ffill

    def create_target(self: