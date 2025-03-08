# Stock Price Prediction System

This repository contains two main components: an ETL and machine learning script for predicting next-day stock price movements, and a Streamlit application that provides real-time predictions and financial overviews for major companies like Tesla (TSLA), Apple (AAPL), Microsoft (MSFT), Nvidia (NVDA), and Meta (META).

## ETL and ML Script

### Overview

The script predicts stock price movements for the next day using historical stock prices. It employs a Decision Tree Classifier, trained to recognize financial data patterns, to perform predictions.

### Data

- **Source**: The dataset is sourced from `Data/us-shareprices-daily.csv`, containing daily share prices.
- **Features**: Includes Open, High, Low, Close, Adjusted Close prices, Trading Volume, Week of the year, and indicators such as Simple Moving Averages and Exponential Moving Averages.

### Model and Scaling

- **Decision Tree Model**: Stored in `DecisionTreeClassifier.pkl`.
- **Scaler**: MinMaxScaler for feature scaling, stored in `scaler_DecisionTree.pkl`.

### Running the Script

The script `main.py` processes data, trains the model, and saves the output without requiring any arguments. Ensure the dataset is available in the `Data` folder before execution.

**Dependencies**: Pandas, Matplotlib, Seaborn, Scikit-learn, Joblib. Install them using:

```bash
pip install pandas matplotlib seaborn scikit-learn joblib
