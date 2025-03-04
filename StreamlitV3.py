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
st.sidebar.title("Navigation 🧭")
page = st.sidebar.radio("Select a Page:", ["Overview 📄", "Predict Next Day 🔮", "Show Financials 📊"])

# Load the CSV file directly into a pandas DataFrame
company_data = pd.read_excel("us-companies exc.xlsx")

if page == "Overview 📄":
    st.title("Project Overview 🌟")
    st.write("""
        ### Project Description 📘
        This project focuses on predicting the next day's stock prices by leveraging historical data. Our approach combines advanced machine learning techniques with robust data processing to ensure accurate predictions.

        ### Tools and Resources 🔧
        - **Python**: Utilized for comprehensive data processing and machine learning tasks.
        - **Streamlit**: Employed to develop an interactive web application that showcases our predictive models.
        - **ChatGPT**: Used as a coding assistant to help debug and optimize our codebase.

        ### Team Overview (Group 8) 👥
        - **Omer Altwaini**: Led the data extraction and ETL processes, ensuring the data integrity needed for effective model training.
        - **James Alarde**: Conducted field research and prepared the datasets for machine learning, bridging the gap between raw data and actionable insights.
        - **Africa Bajils**: Spearheaded the implementation of machine learning algorithms and analyzed data correlations to enhance model accuracy.
        - **Maria do Carmo Brito e Abreu**: Developed the Streamlit application, creating a user-friendly interface that allows users to interact with our predictive models.
        - **Emiliano Puertas**: Responsible for the full documentation and quality assurance, ensuring that all aspects of the project adhere to the highest standards of clarity and accuracy.
    """)

elif page == "Predict Next Day 🔮":
    st.title("Stock Prediction App 🔍")
    ticker = st.selectbox("Select Ticker", ['TSLA', 'AAPL', 'MSFT', 'NVDA', 'META'])
    if st.button("Predict 🔮"):
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

        columns_to_scale = ['Open', 'High', 'Low', 'Close', 'Adj. Close',
           'Volume', 'week', 'First_day',
           'Last_day', 'SMA_7', 'V_SMA_7', 'SMA_14', 'V_SMA_14', 'EMA_7',
           'V_EMA_7', 'EMA_14', 'V_EMA_14']
        
        # Scale data
        X_scaled = scaler.transform(df[columns_to_scale])
        
        # Predict
        prediction = model.predict(X_scaled)
        st.write("Prediction for next day:", "Positive" if prediction[0] else "Negative")

elif page == "Show Financials 📊":
    st.title("Financial Overview 📈")
    ticker = st.selectbox("Select Ticker for Financials", ['TSLA', 'AAPL', 'MSFT', 'NVDA', 'META'])
    company_financials = company_data[company_data['Ticker'] == ticker]
    st.dataframe(company_financials)
