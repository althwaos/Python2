import streamlit as st
import pandas as pd
import joblib
import pickle
import requests
import matplotlib.pyplot as plt
import datetime

def prepare_data(df):
    df.columns = [
        'Date', 'Dividend Paid', 'Common Shares Outstanding',
        'Close', 'Adj. Close', 'High', 'Low', 'Open', 'Volume'
    ]
    df['Date'] = pd.to_datetime(df['Date'])
    df['week'] = df['Date'].dt.isocalendar().week
    df['First_day'] = df['week'] != df['week'].shift(1)
    df['Last_day'] = df['week'] != df['week'].shift(-1)
    df['First_day'].fillna(False, inplace=True)
    df['Last_day'].fillna(False, inplace=True)
    df['SMA_7'] = df['Close'].rolling(window=7).mean()
    df['V_SMA_7'] = df['Volume'].rolling(window=7).mean()
    df['SMA_14'] = df['Close'].rolling(window=14).mean()
    df['V_SMA_14'] = df['Volume'].rolling(window=14).mean()
    df['EMA_7'] = df['Close'].ewm(span=7, adjust=False).mean()
    df['V_EMA_7'] = df['Volume'].ewm(span=7, adjust=False).mean()
    df['EMA_14'] = df['Close'].ewm(span=14, adjust=False).mean()
    df['V_EMA_14'] = df['Volume'].ewm(span=14, adjust=False).mean()
    return df

# Assuming you have a dictionary linking tickers to logo URLs
logo_urls = {
    'TSLA': 'https://example.com/tsla_logo.png',
    'AAPL': 'https://github.com/althwaos/Python2/blob/b5eee693f03c2b117205795aea14f24a55ebe479/Companies%20logos/Apple.png',
    # Add more as needed
}

# Load your model and scaler
model = joblib.load('RandomForestClassifier.pkl')
scaler = pickle.load(open("scaler_RandomForest.pkl", "rb"))

# Load company data
company_data = pd.read_csv('us-companies.csv')

# Streamlit user interface
st.title("Stock Prediction App")
ticker = st.selectbox("Select Ticker", ['TSLA', 'AAPL', 'MSFT', 'NVDA', 'META'])

# Show logo and company details
if ticker in logo_urls:
    st.image(logo_urls[ticker])
    details = company_data[company_data['Ticker'] == ticker]
    st.write(details)  # Displaying the dataframe directly

# Selecting and displaying static company details
static_details = details[['Currency', 'Fiscal Year', 'Total Assets', 'Total Equity']]
st.write(static_details)

# Fetching stock data and predicting
if st.button("Predict Next Day"):
    url = f"https://backend.simfin.com/api/v3/companies/prices/compact?ticker={ticker}&start={pd.Timestamp.today().strftime('%Y-%m-%d')}&end={pd.Timestamp.today().strftime('%Y-%m-%d')}"
    headers = {'Authorization': 'your_api_key'}
    response = requests.get(url, headers=headers)
    data = response.json()
    df = pd.DataFrame(data[0]['data'])
    df = prepare_data(df)  
    columns_to_scale = ['Open', 'High', 'Low', 'Close', 'Adj. Close',
       'Volume', 'week', 'First_day',
       'Last_day', 'SMA_7', 'V_SMA_7', 'SMA_14', 'V_SMA_14', 'EMA_7',
       'V_EMA_7', 'EMA_14', 'V_EMA_14']
    
    # Scale data
    X_scaled = scaler.transform(df[columns_to_scale])
    
    # Predict
    prediction = model.predict(X_scaled)
    next_business_day = (pd.Timestamp.today() + pd.DateOffset(days=1)).strftime('%Y-%m-%d')
    st.write(f"Prediction for {next_business_day}: {'Positive' if prediction[0] else 'Negative'}")

    # Yearly performance graph
    start_date = pd.Timestamp.today() - pd.DateOffset(years=1)
    historical_url = f"https://backend.simfin.com/api/v3/companies/prices/compact?ticker={ticker}&start={start_date.strftime('%Y-%m-%d')}&end={pd.Timestamp.today().strftime('%Y-%m-%d')}"
    hist_response = requests.get(historical_url, headers=headers)
    hist_data = hist_response.json()
    hist_df = pd.DataFrame(hist_data[0]['data'])
    hist_df['Date'] = pd.to_datetime(hist_df['Date'])
    plt.figure(figsize=(10, 5))
    plt.plot(hist_df['Date'], hist_df['Close'], label='Close Price')
    plt.title('Yearly Stock Performance')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)
