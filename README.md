# Stock Price Prediction with Transformers

In this project, we trained a Transformer model to predict a single stock's next day closing price, given a sequence (eg 30 days) of stock data. 

This README comprises of the following sections:
- [Background](https://github.com/mkmuriithi/csc492_deep_learning_project/edit/main/README.md#background)
- [Data Collection](https://github.com/mkmuriithi/csc492_deep_learning_project/edit/main/README.md#data-collection)
- [Transformer Model](https://github.com/mkmuriithi/csc492_deep_learning_project/edit/main/README.md#transformer-model)
- [Results](https://github.com/mkmuriithi/csc492_deep_learning_project/edit/main/README.md#results)
- [Caveats](https://github.com/mkmuriithi/csc492_deep_learning_project/edit/main/README.md#caveats)
- [Ethical Considerations](https://github.com/mkmuriithi/csc492_deep_learning_project/edit/main/README.md#ethical-considerations)
- [Contributing](https://github.com/mkmuriithi/csc492_deep_learning_project/edit/main/README.md#contributing)
- [License](https://github.com/mkmuriithi/csc492_deep_learning_project/edit/main/README.md#license)
---

## Background

### Stock Prediction
Predicting stock prices is one of great interest and reward, yet difficult because they are  unpredictable and volatile.

Statistical and machine learning techniques such as Moving Average, KNN and ARIMA have been commonly used for prediction. 

Recent works tap into deep learning, for its ability to aggregate many information sources and learn patterns. 

### Transformers

A Transformer is a sequence model with self-attention mechanism, meaning it can learn which parts of the input to "attend" or "focus" on.

Unlike other sequence-based architectures eg RNN, a Transformer does not necessarily process the data in order, but provides context for any position in the input sequence.

Transformers have achieved great performance in natural language processing and computer vision. 

We were interested to test its application in time series prediction, in particular stock price prediction.  

If you're not familiar with transformers, first check these out:
- [The Illustrated Transformer by Jay Alammar](https://jalammar.github.io/illustrated-transformer/)
- [Original paper "Attention is All You Need"](https://arxiv.org/abs/1706.03762) 
---

## Data Collection

### Data Sources
We pulled raw stock data, from Yahoo Finance API via the [yfinance](https://pypi.org/project/yfinance/) open source library:
- included 2718 Stocks listed on the NYSE exchange, for data between 2007-01-01 to 2021-12-31 (15 years).
- Taken on a 1 day interval.   
- About 50% of the stocks have the full timeframe of data
- The remainder had data starting from the date that they were listed.
- About 20% of stocks are less than 3 years old, and were discarded due to insufficient time frame.

We performed data cleaning to remove securities, and drop stocks that had missing data.

The raw data is not included in this repo, but you can obtain it by requesting data from yfinance.

### Data representation
#### Features for a given day's data
A given day of a stock’s history is a vector of 10 features. 

5 features are pulled from yfinance API,  and other 5 are additional engineered features commonly used for detecting momentum in stock price.

Basic Features:
- Opening Price
- High (Highest stock price 
reached during the day)
- Low (Lowest stock price 
reached during the day)
- Adjusted Closing Price
- Volume

Engineered Features:
- One-hot Day-of-week 
(1 feature per weekday)
- 10-Day Simple Moving Average
- 10-Day Exponential Moving Average
- Relative Strength Index
- Volume Weighted Average Price
<img width="457" alt="Screenshot 2022-04-18 at 7 20 48 AM" src="https://user-images.githubusercontent.com/17236572/163821812-e3cd0396-c64f-4859-b7c8-6d01a3d2ce3e.png">


#### Data point sequence of 30 day history

We define one data point as a sequence of 30 days of stock history.

Each batch of data points is normalised by:
- Converting features to percentage change from previous day
- Min-max scaling relative to the entire training set

### Data Split
Each stock data was split into training, validation, and test sets by date with a ratio of 80:10:10. 

Stocks with less than 3 years of data were excluded to ensure sufficient data for each set.
<img width="471" alt="Screenshot 2022-04-18 at 7 00 12 AM" src="https://user-images.githubusercontent.com/17236572/163820977-1f840976-8694-4e42-8480-e703090f4563.png">

---

## Model
We decided to evaluate two types of model:
- Single-stock: Model is intended for use on a specific stock; the model is trained on data from only that stock 
- Multi-stock: Model is intended for use on any stock; the model is trained on the entire training set of all stock data

### Model Architecture
- We first embed the input features using a linear layer followed by positional encoding.
- We then pass data through 4 transformer encoder layers, as described in the paper “Attention Is All You Need”.
- Next, we take the maximum and mean of each feature over the output sequence.
- Finally, we use a linear layer to combine the maximum and mean values into a single prediction.
<img width="152" alt="Screenshot 2022-04-18 at 7 05 09 AM" src="https://user-images.githubusercontent.com/17236572/163820187-9ef23a55-645c-4bb4-9793-7dc2478296b2.png">
---

## Results


---
## Caveats
- The training data only consists of companies that still exist in 2021, and excludes companies that folded up before that. This leads to survivorship bias favouring companies that succeed.
- The model is trained purely on stock data, which leaves out other factors such as market, economy and the company.

---
## Ethical Considerations
- Autonomy: Used publicly available, open source APIs to ensure consent for data collection
- Beneficence: This model may benefit people to capitalise on predictions, while risking others who do not understand the ML techniques
- Justice: This model may democratise trading by making predictions available to the public
- Non-maleficence: In the event of widespread adoption of a model's predictions, it may affect traders' behaviour, stock demand, and stock prices of companies. However, this likely to only affect small market cap companies.
---

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)
