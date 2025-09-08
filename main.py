# import 
import os
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
import numpy as np

#display max columns
pd.set_option('display.max_columns', None)


#pull data 
dataPath = "stock_data.json"

if os.path.exists(dataPath):
    with open(dataPath) as f:
        stockHist = pd.read_json(dataPath)
else:
    stock = yf.Ticker("GOOG")
    stockHist = stock.history(period="max")
    stockHist.to_json(dataPath)

#get actual close
data = stockHist[["Close"]]
data = data.rename(columns = {'Close':'Actual_Close'})

#get the target (1/0)
data["Target"] = stockHist.rolling(2).apply(lambda x: x.iloc[1] > x.iloc[0])["Close"]

#copy to new dataframe and shift data down 1 row
stockPrev = stockHist.copy()
stockPrev = stockPrev.shift(1)

#predictors and combine data
predictors = ["Close", "Volume", "Open", "High", "Low"]
data = data.join(stockPrev[predictors]).iloc[1:]

weekly_mean = data.rolling(7).mean()["Close"]
quarterly_mean = data.rolling(90).mean()["Close"]
annual_mean = data.rolling(365).mean()["Close"]

weekly_trend = data.shift(1).rolling(7).sum()["Target"]

data["weekly_mean"] = weekly_mean / data["Close"]
data["quarterly_mean"] = quarterly_mean / data["Close"]
data["annual_mean"] = annual_mean / data["Close"]

data["annual_weekly_mean"] = data["annual_mean"] / data["weekly_mean"]
data["annual_quarterly_mean"] = data["annual_mean"] / data["quarterly_mean"]

data["weekly_trend"] = weekly_trend

data["open_close_ratio"] = data["Open"] / data["Close"]
data["high_close_ratio"] = data["High"] / data["Close"]
data["low_close_ratio"] = data["Low"] / data["Close"]

fullPredictors = predictors + ["weekly_mean", "quarterly_mean", "annual_mean", "annual_weekly_mean", "annual_quarterly_mean", "open_close_ratio", "high_close_ratio","low_close_ratio"]


model = RandomForestClassifier(n_estimators=1000, min_samples_split=2000, random_state=1)

#model

def backtest(data, model, predictors, start=1000, step=50):
    predictions = []
    # Loop over the dataset in increments
    for i in range(start, data.shape[0], step):
        # Split into train and test sets
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i+step)].copy()

        # Fit the random forest model
        model.fit(train[predictors], train["Target"])

        # Make predictions
        preds = model.predict_proba(test[predictors])[:,1]
        preds = pd.Series(preds, index=test.index)
        preds[preds > .6] = 1
        preds[preds<=.6] = 0

        # Combine predictions and test values
        combined = pd.concat({"Target": test["Target"],"Predictions": preds}, axis=1)

        predictions.append(combined)

    return pd.concat(predictions)

predictions = backtest(data.iloc[365:], model, fullPredictors)
predictions["Predictions"].value_counts()
predictions["Target"].value_counts()
print(precision_score(predictions["Target"], predictions["Predictions"]))
predictions.iloc[-100:].plot()
plt.show()