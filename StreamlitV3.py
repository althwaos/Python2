import streamlit as st
import pandas as pd
import joblib
import pickle
import requests
import matplotlib.pyplot as plt

# Load your model and scaler
model = joblib.load('RandomForestClassifier.pkl')
scaler = pickle.load(open("scaler_RandomForest.pkl", "rb"))

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

# Define the page layout
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a Page:", ["Overview", "Predict Next Day", "Show Financials"])

# Load the CSV file directly into a pandas DataFrame
company_data = pd.read_excel("us-companies exc.xlsx")

if page == "Overview":
    st.title("Project Overview")
    st.write("""
        ### Description
        This project aims to predict next day stock prices using historical data.
        
        ### Resources Used
        - Python for data processing and machine learning.
        - Streamlit for creating the web application.
        - Pandas for data manipulation.
        
        ### Team Overview
        - **Alice**: Data Scientist - Responsible for model building and testing.
        - **Bob**: Frontend Developer - Managed the Streamlit dashboard.
        - **Charlie**: Data Engineer - Handled data collection and preprocessing.
    """)

elif page == "Predict Next Day":
    st.title("Stock Prediction App")
    ticker = st.selectbox("Select Ticker", ['TSLA', 'AAPL', 'MSFT', 'NVDA', 'META'])
    if st.button("Predict"):
        url = f"https://backend.simfin.com/api/v3/companies/prices/compact?ticker={ticker}&start=2025-01-01&end=2025-06-01"
        headers = {'Authorization': 'Your_API_Key'}
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

        columns_to_scale = ['Open', 'High', 'Low', 'Close', 'Adj. Close', 'Volume', 'week', 'First_day', 'Last_day', 'SMA_7', 'V_SMA_7', 'SMA_14', 'V_SMA_14', 'EMA_7', 'V_EMA_7', 'EMA_14', 'V_EMA_14']
        X_scaled = scaler.transform(df[columns_to_scale])
        prediction = model.predict(X_scaled)
        st.write("Prediction for next day:", "Positive" if prediction[0] else "Negative")

elif page == "Show Financials":
    st.title("Financial Overview")
    ticker = st.selectbox("Select Ticker for Financials", ['TSLA', 'AAPL', 'MSFT', 'NVDA', 'META'])
    company_financials = company_data[company_data['Ticker'] == ticker]
    st.dataframe(company_financials)
