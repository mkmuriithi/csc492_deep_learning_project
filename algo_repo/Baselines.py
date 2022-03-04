import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression


def train_previous_price(df_allstocks):
    avg_mse = 0
    avg_rmse = 0
    avg_mape = 0
    for stock in df_allstocks.keys():

        try:
            df_stock = df_allstocks[stock]
            if "y" not in list(df_stock.columns):
                df_stock = make_supervised(df_stock)
            df_stock.dropna(inplace=True)
            X = df_stock.drop(["y", 'ticker'], axis=1)
            y = df_stock["y"].copy(deep=True)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
            df_results = pd.DataFrame({"Real Closing Price": X_test.Close, "Predicted Closing": y_test}).dropna()
            mse = mean_squared_error(df_results["Real Closing Price"], df_results["Predicted Closing"])
            avg_mse += mse
            avg_rmse += np.sqrt(mse)
            avg_mape += mean_absolute_percentage_error(df_results["Real Closing Price"], df_results["Predicted Closing"])
        except Exception as e:
            print(f'Ticker {stock} caused problem: {e}')
            continue
    avg_mse /= len(df_allstocks.keys())
    avg_rmse /= len(df_allstocks.keys())
    avg_mape /= len(df_allstocks.keys())

    return avg_mse, avg_rmse, avg_mape


def train_linear(df_allstocks):
    avg_mse = 0
    avg_rmse = 0
    avg_mape = 0
    count = 0
    for stock in df_allstocks.keys():
        try:
            count += 1
            df_stock = df_allstocks[stock]
            if "y" not in list(df_stock.columns):
                df_stock = make_supervised(df_stock)
            df_stock.dropna(inplace=True)
            X = df_stock.drop(["y", "ticker", "Date"], axis=1)
            y = df_stock["y"].copy(deep=True)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

            ols = LinearRegression(n_jobs=4)
            ols.fit(X_train, y_train)
            predictions = ols.predict(X_test)

            mse = mean_squared_error(y_test, predictions)
            avg_mse += mse
            avg_rmse += np.sqrt(mse)
            avg_mape += mean_absolute_percentage_error(y_test, predictions)

            if count % 100 == 0:
                print(f'Iteration {count} completed')
        except Exception as e:
            print(f'Ticker {stock} caused problem: {e}')
            continue

    avg_mse /= len(df_allstocks.keys())
    avg_rmse /= len(df_allstocks.keys())
    avg_mape /= len(df_allstocks.keys())

    return avg_mse, avg_rmse, avg_mape


def make_supervised(df):
    # make dataset into supervised problem
    df["y"] = df["Close"].shift(periods=1)
    return df
