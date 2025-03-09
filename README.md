# Stock Price Prediction System

This repository contains two main components: an ETL and machine learning script for predicting next-day stock price movements, and a Streamlit application that provides real-time predictions and financial overviews for major companies like Tesla (TSLA), Apple (AAPL), Microsoft (MSFT), Nvidia (NVDA), and Meta (META).

https://python2-a6dqddvgf8wjdwukaefyrs.streamlit.app/

## ETL and ML Script

### Overview

The script predicts stock price movements for the next day using historical stock prices. It employs a Decision Tree Classifier, trained to recognize financial data patterns, to perform predictions.

### Data

- **Source**: The dataset is sourced from `Data/us-shareprices-daily.csv`, containing daily share prices.
- **Features**: Includes Open, High, Low, Close, Adjusted Close prices, Trading Volume, Week of the year, and indicators such as Simple Moving Averages and Exponential Moving Averages.

### Data preparation

**After reseraching the stock market we found out that the following details were important to add in order to predict next day price action:**
- Simple Moving Average (SMA) for closing price in both 7 days and 14 days window.
- Exponential Moving Average (EMA) for closing price in both 7 days and 14 days window.
- Simple Moving Average (SMA) for volumne in both 7 days and 14 days window.
- Exponential Moving Average (EMA) for volumne  in both 7 days and 14 days window.
- First Day of the trading week or last day
  
### Model and Scaling

- **Decision Tree Model**: Stored in `DecisionTreeClassifier.pkl`.
We have selected the decision tree model becuase in our approach we developed one model that is able to interact differently depedning on the ticker, and for that we needed to go with the Decision Tree Model
- **Scaler**: MinMaxScaler for feature scaling, stored in `scaler_DecisionTree.pkl`.

### Running the Script

The script `main.py` processes data, trains the model, and saves the output without requiring any arguments. Ensure that this 'us-shareprices-daily.csv' data set is downloaded in your 'Data Folder' before execution.


## Streamlit notebook

### Overview

The script is basically reciving an API call depedning on the user selection of the 5 availble tickers, and then process the data in-spot, implement the exisitng ML developed model and the scalar to predict next day price. Additionaly, we have added some more user-firendly interactions such as emojis and a lot of charts for better user interaction with the website. 

### Data

- **Source**: One dataset is an API retriving data of daily stock market for the sleceted ticker. Another two datasets, are batch data sets and they are used to show more details about the company.
  
### Data preparation

Aliging with the same process we have done in the ETL process, however we are doing it here for the API call. 
  
### Sections of the website

- **Overview**: In this page we are showing overview of the project, tools we used for the project, steps to complete the project, and the team members and responsibilities.
- **Predict Next Day**: Select a ticker, predict next day if positive or negative, and give some insights in a chart of how the price was closing for the last 90 days.
- **Show Financials**: Show some details about the selected company, and the total assets, liabilities, and equity for last 12 quarters. 




