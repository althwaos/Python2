# Metting Porject deliveries requirements:
✅ Presentation 10-15 miutes (attached in the blackboard)
✅ `main.py` scrpipt that auto-generate the ML model
✅ Streamlit application
✅ Excutive summary
✅ README.md file

# Stock Price Prediction System

This repository contains two main components: an ETL and machine learning script for predicting next-day stock price movements, and a Streamlit application that provides real-time predictions and financial overviews for major companies like Tesla (TSLA), Apple (AAPL), Microsoft (MSFT), Nvidia (NVDA), and Meta (META).

https://python2-a6dqddvgf8wjdwukaefyrs.streamlit.app/

## GitHub files
You will find a lot of files in this project, the reason behind it is the way we had our project execution was delivered in steps: 
- Dividing the workload to all team members
- Each member was responsible for their own notebook
- After developing these notebooks (ETL, ML, API calls, and Streamlit) and making sure that they worked well-together
- Then we combined the ETL process and ML in one script, and connected the streamlit app to the script outcome and made the API calls within the streamlit app

In  `Project Time Line (Python).xlsx`  file you may find all the details of workload distribution. 

## ETL and ML Script

### Overview

The script predicts stock price movements for the next day using historical stock prices. It employs a Decision Tree Classifier, trained to recognize financial data patterns, to perform predictions.

### Data

- **Source**: The dataset is sourced from `Data/us-shareprices-daily.csv`, containing daily share prices.
- **Features**: Includes Open, High, Low, Close, Adjusted Close prices, Trading Volume, Week of the year, and indicators such as Simple Moving Averages and Exponential Moving Averages.

### Data preparation

**After researching the stock market we found out that the following details were important to add in order to predict next day price action:**
- Simple Moving Average (SMA) for closing price in both 7 days and 14 days window.
- Exponential Moving Average (EMA) for closing price in both 7 days and 14 days window.
- Simple Moving Average (SMA) for volume in both 7 days and 14 days window.
- Exponential Moving Average (EMA) for volume in both 7 days and 14 days window.
- First Day of the trading week or last day
  
### Model and Scaling

- **Decision Tree Model**: Stored in `DecisionTreeClassifier.pkl`.
We have selected the decision tree model because in our approach we developed one model that is able to interact differently depending on the ticker, and for that we needed to go with the Decision Tree Model
- **Scaler**: MinMaxScaler for feature scaling, stored in `scaler_DecisionTree.pkl`.

### Running the Script

The script `main.py` processes data, trains the model, and saves the output without requiring any arguments. Ensure that this `us-shareprices-daily.csv` data set is downloaded in your `Data Folder` before execution.


## Streamlit notebook

### Overview

The script is basically receiving an API call depending on the user selection of the 5 available tickers, and then process the data in-spot, implement the existing ML developed model and the scalar to predict next day price. Additionaly, we have added some more user-friendly interactions such as emojis and a lot of charts for better user interaction with the website. 

### Data

- **API**: API retriving data of daily stock market for the sleceted ticker.
-**Batches Data**: Another two datasets, are companies details and balance sheets. 
  
### Data preparation

Aligning with the same process we have done in the ETL process, however we are doing it here for the API call. 
  
### Sections of the website

- **Overview**: In this page we are showing overview of the project, tools we used for the project, steps to complete the project, and the team members and responsibilities.
- **Predict Next Day**: Select a ticker, predict next day if positive or negative, and give some insights in a chart of how the price was closing for the last 90 days. Additionaly we have added here the suggested trading strategy with a simulation of usage with a 10,000$
- **Show Financials**: Show some details about the selected company, and the total assets, liabilities, and equity for last 12 quarters. 




