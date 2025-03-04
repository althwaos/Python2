import streamlit as st
import pandas as pd
import joblib
import pickle
import requests
import matplotlib.pyplot as plt


# Define a function to prepare data
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

# Load your model and scaler
model = joblib.load('RandomForestClassifier.pkl')
scaler = pickle.load(open("scaler_RandomForest.pkl", "rb"))

logo_urls = {
    'TSLA': 'https://raw.githubusercontent.com/your_username/your_repo/main/logos/tsla.png',
    'AAPL': 'https://github.com/althwaos/Python2/blob/main/Apple.png',
    'MSFT': 'https://raw.githubusercontent.com/your_username/your_repo/main/logos/msft.png',
    'NVDA': 'https://raw.githubusercontent.com/your_username/your_repo/main/logos/nvda.png',
    'META': 'https://raw.githubusercontent.com/your_username/your_repo/main/logos/meta.png',
}

# Load the CSV file directly into a pandas DataFrame
company_data = pd.read_excel("us-companies exc.xlsx")


# Streamlit user interface
st.title("Stock Prediction App")
ticker = st.selectbox("Select Ticker", ['TSLA', 'AAPL', 'MSFT', 'NVDA', 'META'])

if ticker in logo_urls:
    st.image(logo_urls[ticker], width=200)

if st.button("Predict Next Day"):
    url = f"https://backend.simfin.com/api/v3/companies/prices/compact?ticker={ticker}&start=2025-01-01&end=2025-06-01"
    headers = {'Authorization': 'fde00d2f-38ad-43f3-9a83-012077d42da6'}
    response = requests.get(url, headers=headers)
    data = response.json()
    df = pd.DataFrame(data[0]['data'])
    df = prepare_data(df)  
    
    plt.figure(figsize=(10, 4))
    plt.plot(pd.to_datetime(df['Date']), df['Close'], label='Close Price')
    plt.title('Close Price by Day for the Last Year')
    plt.xlabel('Date')
    plt.ylabel('Close Price ($)')
    plt.legend()
    st.pyplot(plt)

    # Assuming 'columns_to_scale' is defined elsewhere in your script
    columns_to_scale = ['Open', 'High', 'Low', 'Close', 'Adj. Close',
       'Volume', 'week', 'First_day',
       'Last_day', 'SMA_7', 'V_SMA_7', 'SMA_14', 'V_SMA_14', 'EMA_7',
       'V_EMA_7', 'EMA_14', 'V_EMA_14']
    
    # Scale data
    X_scaled = scaler.transform(df[columns_to_scale])
    
    # Predict
    prediction = model.predict(X_scaled)
    st.write("Prediction for next day:", "Positive" if prediction[0] else "Negative")

if st.button("Show Financials"):
    # Filter data for the selected company
    company_financials = company_data[company_data['Ticker'] == ticker]
    st.dataframe(company_financials) 
