from __future__ import (absolute_import, division, print_function, unicode_literals)

import backtrader as bt
import datetime
import os.path
import sys

#Test Strategy
class TestStrategy(bt.Strategy): #every strategy must inherit this bt.Strategy

    def log(self, txt, dt=None):
        #logging function
        dt = dt or self.datas[0].datetime.date(0)
        print("{}, {}".format(dt.isoformat(), txt))
    def __init__(self):
        #reference to "close" line in the data[0] dataseries
        self.dataclose = self.datas[0].close
        self.dataopen = self.datas[0].open
        self.volume = self.datas[0].volume
    def next(self):
        #log closing price of series from the reference
        self.log('Open: {}, Close: {}, Volume: {}'.format(self.dataopen[0],self.dataclose[0], self.volume[0]))
        if self.dataclose[0] < self.dataclose[-1] and self.dataclose[-1] < self.dataclose[-2]:
            self.log("Buy at {}".format(self.dataclose[0]))
            self.buy()




if __name__ == '__main__':

    
    cerebro = bt.Cerebro()
    
    #add strategy
    cerebro.addstrategy(TestStrategy)
    data_path = "./GOOG.csv"
    data = bt.feeds.YahooFinanceCSVData(dataname=data_path, fromdate = datetime.datetime(2016, 1, 1), 
            todate = datetime.datetime(2021, 12, 30),
            reverse=False)
    cerebro.adddata(data)
    cerebro.broker.setcash(100000)

    print('Starting Portfolio Value: {}'.format(cerebro.broker.getvalue()))
    cerebro.run()
    print('Final Portfolio Value: {}'.format(cerebro.broker.getvalue()))
