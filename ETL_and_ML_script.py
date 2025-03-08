import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import pickle
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    try:
        # Load data
        logging.info("Loading data...")
        stock = pd.read_csv('Data/us-shareprices-daily.csv', sep=";")
        stock['Date'] = pd.to_datetime(stock['Date'])
        stock = stock[stock['Ticker'].isin(['TSLA', 'AAPL', 'MSFT', 'NVDA', 'META'])]
        
        # Prepare data
        logging.info("Preparing data...")
        stock['week'] = stock['Date'].dt.isocalendar().week
        stock['First_day'] = stock['week'] != stock['week'].shift(1)
        stock['Last_day'] = stock['week'] != stock['week'].shift(-1)
        stock.fillna(False, inplace=True)
        stock['SMA_7'] = stock['Close'].rolling(window=7).mean()
        stock['V_SMA_7'] = stock['Volume'].rolling(window=7).mean()
        stock['SMA_14'] = stock['Close'].rolling(window=14).mean()
        stock['V_SMA_14'] = stock['Volume'].rolling(window=14).mean()
        stock['EMA_7'] = stock['Close'].ewm(span=7, adjust=False).mean()
        stock['V_EMA_7'] = stock['Volume'].ewm(span=7, adjust=False).mean()
        stock['EMA_14'] = stock['Close'].ewm(span=14, adjust=False).mean()
        stock['V_EMA_14'] = stock['Volume'].ewm(span=14, adjust=False).mean()
        
        # Define target
        stock['Tomorrow_Close_Positive'] = stock.groupby('Ticker')['Close'].transform(lambda x: x.shift(-1) > x)
        stock.dropna(inplace=True)

        # Ticker encoding
        ticker_mapping = {'AAPL': 1, 'META': 2, 'MSFT': 3, 'NVDA': 4, 'TSLA': 5}
        stock['Ticker_cat'] = stock['Ticker'].map(ticker_mapping)

        # Feature selection
        features = ['Ticker_cat', 'Open', 'High', 'Low', 'Close', 'Adj. Close',
       'Volume', 'week', 'First_day',
       'Last_day', 'SMA_7', 'V_SMA_7', 'SMA_14', 'V_SMA_14', 'EMA_7',
       'V_EMA_7', 'EMA_14', 'V_EMA_14']
        X = stock[features]
        y = stock['Tomorrow_Close_Positive']

        # Data Splitting
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2001)
        columns_to_scale = ['Ticker_cat', 'Open', 'High', 'Low', 'Close', 'Adj. Close',
       'Volume', 'week', 'First_day',
       'Last_day', 'SMA_7', 'V_SMA_7', 'SMA_14', 'V_SMA_14', 'EMA_7',
       'V_EMA_7', 'EMA_14', 'V_EMA_14']
        # Feature Scaling
        scaler = MinMaxScaler()
        scaler.fit(X_train[columns_to_scale])
        X_train_scaled = X_train  # Avoid modifying the original DataFrame
        X_test_scaled = X_test

        X_train_scaled[columns_to_scale] = scaler.transform(X_train[columns_to_scale])
        X_test_scaled[columns_to_scale] = scaler.transform(X_test[columns_to_scale])

        # Model Training
        model = DecisionTreeClassifier(random_state=2001)
        model.fit(X_train_scaled, y_train)

        # Save model and scaler
        joblib.dump(model, 'DecisionTreeClassifier.pkl')
        pickle.dump(scaler, open("scaler_DecisionTree.pkl", "wb"))
        
        # Log success
        logging.info("Model and scaler saved successfully.")
        
    except FileNotFoundError:
        logging.error("Data file not found. Please check the file path and try again.")
    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == '__main__':
    main()
