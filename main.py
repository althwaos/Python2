import pandas as pd
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
import joblib
import pickle

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class StockPricePredictor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.model = None
        self.scaler = None
        self.stock = None
        self.features = ['Ticker_cat', 'Open', 'High', 'Low', 'Close', 'Adj. Close',
                         'Volume', 'week', 'First_day', 'Last_day', 'SMA_7', 'V_SMA_7',
                         'SMA_14', 'V_SMA_14', 'EMA_7', 'V_EMA_7', 'EMA_14', 'V_EMA_14']

    def load_data(self):
        try:
            self.stock = pd.read_csv(self.data_path, sep=";")
            self.stock['Date'] = pd.to_datetime(self.stock['Date'])
            self.stock = self.stock[self.stock['Ticker'].isin(['TSLA', 'AAPL', 'MSFT', 'NVDA', 'META'])]
            logging.info("Data loaded successfully.")
        except FileNotFoundError:
            logging.error("Data file not found. Please check the file path and try again.")
            raise

    def prepare_data(self):
        logging.info("Preparing data...")
        self.stock['week'] = self.stock['Date'].dt.isocalendar().week
        self.stock['First_day'] = self.stock['week'] != self.stock['week'].shift(1)
        self.stock['Last_day'] = self.stock['week'] != self.stock['week'].shift(-1)
        self.stock.fillna(False, inplace=True)
        self.calculate_features()
        ticker_mapping = {'AAPL': 1, 'META': 2, 'MSFT': 3, 'NVDA': 4, 'TSLA': 5}
        self.stock['Ticker_cat'] = self.stock['Ticker'].map(ticker_mapping)
        self.stock['Tomorrow_Close_Positive'] = self.stock.groupby('Ticker')['Close'].transform(lambda x: x.shift(-1) > x)
        self.stock.dropna(inplace=True)

    def calculate_features(self):
        self.stock['SMA_7'] = self.stock['Close'].rolling(window=7).mean()
        self.stock['V_SMA_7'] = self.stock['Volume'].rolling(window=7).mean()
        self.stock['SMA_14'] = self.stock['Close'].rolling(window=14).mean()
        self.stock['V_SMA_14'] = self.stock['Volume'].rolling(window=14).mean()
        self.stock['EMA_7'] = self.stock['Close'].ewm(span=7, adjust=False).mean()
        self.stock['V_EMA_7'] = self.stock['Volume'].ewm(span=7, adjust=False).mean()
        self.stock['EMA_14'] = self.stock['Close'].ewm(span=14, adjust=False).mean()
        self.stock['V_EMA_14'] = self.stock['Volume'].ewm(span=14, adjust=False).mean()
        
    def train_model(self):
        X = self.stock[self.features]
        y = self.stock['Tomorrow_Close_Positive']
        columns_to_scale = ['Ticker_cat', 'Open', 'High', 'Low', 'Close', 'Adj. Close',
       'Volume', 'week', 'First_day',
       'Last_day', 'SMA_7', 'V_SMA_7', 'SMA_14', 'V_SMA_14', 'EMA_7',
       'V_EMA_7', 'EMA_14', 'V_EMA_14']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2001)
        self.scaler = MinMaxScaler()
        X_train[columns_to_scale] = self.scaler.fit_transform(X_train)
        X_test[columns_to_scale] = self.scaler.transform(X_test)

        self.model = DecisionTreeClassifier(random_state=2001)
        self.model.fit(X_train, y_train)
        logging.info("Model trained successfully.")

    def save_model(self):
        joblib.dump(self.model, 'DecisionTreeClassifier.pkl')
        pickle.dump(self.scaler, open("scaler_DecisionTree.pkl", "wb"))
        logging.info("Model and scaler saved successfully.")

def main():
    predictor = StockPricePredictor('Data/us-shareprices-daily.csv')
    predictor.load_data()
    predictor.prepare_data()
    predictor.train_model()
    predictor.save_model()

if __name__ == '__main__':
    main()
