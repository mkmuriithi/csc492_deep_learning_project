from __future__ import (absolute_import, division, print_function, unicode_literals)
import os
import backtrader as bt
import datetime
import os.path
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pytz
from qstrader.statistics.tearsheet import TearsheetStatistics
from math import floor
TSLApath='TSLA.csv'
GOOGpath='./GOOG.csv'

fromdate = datetime.datetime(2000,1,1) #don't pass values before this date
todate=datetime.datetime(2000,12,31) #don't pass values after that date

#creating data feed
data = bt.feeds.YahooFinanceCSVData(dataname=TSLApath,fromdate=fromdate,todate=todate,reverse=False)

#strategy_equity_curve = pd.DataFrame(columns=["date", "Equity"]) #for tearsheet
#strategy_equity_curve = {'data': [], 'Equity': []}
#benchmark_equity_curve = pd.DataFrame(columns=["date", "Equity"]) #for tearsheet
#benchmark_equity_curve = {'data': [], 'Equity': []}


class TestStrategy(bt.Strategy): #every strategy must inherit this bt.Strategy
    
    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print("{}, {}".format(dt.isoformat(), txt))

        #add to the strategy_equity_curve_pandas
        
    def __init__(self):
        self.dataclose = self.datas[0].close
        self.dataopen = self.datas[0].open
        self.volume = self.datas[0].volume
        self.order = None
        self.buyprice = None
        self.buycomm = None
        self.bar_executed = len(self)
        self.strategy_equity_curve = pd.DataFrame(columns=["date", "Equity"]) 
        self.last_date = datetime.datetime(2021,5,24)#need this so we know when to write to csv
        #self.sma = bt.indicators.MovingAverageSimple(self.datas[0], period=self.params.mapperiod)
        #plotting indicators
        #bt.indicators.ExponentialMovingAverage(self.datas[0], period=25)
        #bt.indicators.WeightedMovingAverage(self.datas[0], period=25,subplot=True)
        #bt.indicators.StochasticSlow(self.datas[0])
        #bt.indicators.MACDHisto(self.datas[0])
        #rsi = bt.indicators.RSI(self.datas[0])
        #bt.indicators.SmoothedMovingAverage(rsi, period=10)
        #bt.indicators.ATR(self.datas[0], plot=False)
        
    def notify_order(self, order):
        #check if the order status is the same as Submitted or Accepted statuses
        if (order.status in [order.Submitted, order.Accepted]):
            #then continue, nothing to do here since broker still has our order
            return
        elif (order.status in [order.Completed]): #means order was compeleted
            #we want to check if its a buy or sell order
            if order.isbuy():
                #log it
                self.log("Buy completed at price {} with commission {}".format(order.executed.price, order.executed.comm))
                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            else:
                #its a sell
                self.log("Sell order completed")
            self.bar_executed = len(self)
            self.log("The length of len(self) is currently {}".format(len(self)))
        self.order = None #after you confirm order is completed or cancelled, set this to None as no pending orders
        
    def notify_trade(self, trade):
        
        #check if a position has recently been closed (liquidated)
        if(trade.isclosed):
            #then print how much my pnl is
            self.log("Current gross profit is {} and net is {}".format(trade.pnl, trade.pnlcomm))
            
    def next(self):
        #log closing price of series from the reference
        self.log(f'Open: {self.dataopen[0]}, Close: {self.dataclose[0]}, Volume: {self.volume[0]}')
        
        #check if there is an open order, if there is return
        if self.order:
            return
        
        #check if we are in the market. If no, consider buying, if yes consider selling
        if not self.position:
            if self.dataclose[0] < self.dataclose[-1] and self.dataclose[-1] < self.dataclose[-2]:
                self.log("Buy at {}".format(self.dataclose[0]))
                self.buy(size=1)
        #else: #are in the market    
            #if len(self) >= (self.bar_executed + 5): #its been more than 5 days since our position
                #self.log("SELL ORDER CREATED at {}".format(self.dataclose[0]))
                #self.sell()
        
        date_to_add = self.datas[0].datetime.date(0)
        equity_to_add = self.broker.getvalue()
        self.strategy_equity_curve = self.strategy_equity_curve.append(pd.DataFrame({'date': [date_to_add], 'Equity': [equity_to_add]}))
        backtrader_date = self.datas[0].datetime.date()
        backtrader_date_year = backtrader_date.year
        backtrader_date_month = backtrader_date.month
        backtrader_date_day = backtrader_date.day

        ending_date = self.last_date
        ending_date_year = ending_date.year
        ending_date_month = ending_date.month
        ending_date_day = ending_date.day

        

        if backtrader_date_year == ending_date_year and backtrader_date_month == ending_date_month and backtrader_date_day == ending_date_day:
            self.strategy_equity_curve.to_csv("strategy_equity_curve.csv")
                #print("This is where we have that print line\n with len(self) is {} and bar_executed is {}".format(len(self), self.bar_executed))


class BenchmarkStrategy(bt.Strategy):
    #simply a buy and hold strategy
    
    def __init__(self, fromdate=None, todate=None, data=None):
        self.dataclose = self.datas[0].close
        self.dataopen = self.datas[0].open
        self.volume = self.datas[0].volume
        self.order = None
        self.buyprice = None
        self.buycomm = None
        self.bar_executed = len(self)
        self.benchmark_equity_curve = pd.DataFrame(columns=["date", "Equity"]) 
        self.last_date = datetime.datetime(2021,5,24)#need this so we know when to write to csv

    def log(self, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        #log the date and the current portfolio value

    def next(self):
        #our entire strategy is right from day one, we will put all our money into the S%P 500
        #as soon as we have enough money to afford more, we will buy more

        current_price = self.dataclose[0]
        current_cash = self.broker.get_cash()

        #see if we can afford
        if current_cash / current_price > 1:
            amount_to_buy = floor(current_cash / current_price)
            self.buy(size=amount_to_buy)

        date_to_add = self.datas[0].datetime.date(0)
        equity_to_add = self.broker.getvalue()
        self.benchmark_equity_curve = self.benchmark_equity_curve.append(pd.DataFrame({'date': [date_to_add], 'Equity': [equity_to_add]}))

        backtrader_date = self.datas[0].datetime.date()
        backtrader_date_year = backtrader_date.year
        backtrader_date_month = backtrader_date.month
        backtrader_date_day = backtrader_date.day

        ending_date = self.last_date
        ending_date_year = ending_date.year
        ending_date_month = ending_date.month
        ending_date_day = ending_date.day

        

        if backtrader_date_year == ending_date_year and backtrader_date_month == ending_date_month and backtrader_date_day == ending_date_day:
            self.benchmark_equity_curve.to_csv("benchmark_equity_curve.csv")
        self.log()
            
    
    



   # def log(self, txt, dt=None):
        #log the whole frame into 
if __name__ == '__main__':
    
    cerebro=bt.Cerebro() #instantiating cerebro engine
    cerebro.addstrategy(TestStrategy) #adding our strategy to the engine
    data_path = "./GOOG.csv"
    #data = bt.feeds.YahooFinanceCSVData(dataname=data_path, fromdate = datetime.datetime(2016, 1, 1),
     #                               todate = datetime.datetime(2021, 12, 30),reverse=False)
    data = bt.feeds.GenericCSVData(dataname=data_path, tmformat=None,fromdate=datetime.datetime(2016,1,1),
            todate=datetime.datetime(2021, 12, 30), reverse=False)
    
    cerebro.adddata(data)
    cerebro.broker.setcash(100000)
    cerebro.broker.setcommission(commission=0.002)
    #cerebro.addsizer(bt.sizers.FixedSize, stake=10)


    print('Starting Portfolio Value: {}'.format(cerebro.broker.getvalue()))
    cerebro.run()
    print('Final Portfolio Value: {}'.format(cerebro.broker.getvalue()))
    #cerebro.plot()i

    #do benchmarking strategy
    cerebro=bt.Cerebro()
    cerebro.addstrategy(BenchmarkStrategy) #adding our strategy to the engine
    data_path = "./GOOG.csv"
    data = bt.feeds.YahooFinanceCSVData(dataname=data_path, fromdate = datetime.datetime(2016, 1, 1),
                                    todate = datetime.datetime(2021, 12, 30),reverse=False)
    cerebro.adddata(data)
    cerebro.broker.setcash(100000)
    cerebro.broker.setcommission(commission=0.002)

    cerebro.run()

    #Create tearsheet
    #read equity curves form csv
    strategy = pd.read_csv('strategy_equity_curve.csv')
    benchmark = pd.read_csv('benchmark_equity_curve.csv')
    strategy['date'] = pd.to_datetime(strategy['date'])
    benchmark['date'] = pd.to_datetime(benchmark['date'])
    strategy = strategy[['date', 'Equity']].set_index('date')
    benchmark = benchmark[['date', 'Equity']].set_index('date')

    tearsheet = TearsheetStatistics(
        strategy_equity=strategy,
        benchmark_equity=benchmark,
        title='First Strategy'
    )
    tearsheet.plot_results()

    if __name__ == '__main__':
        print("run the stuff")

