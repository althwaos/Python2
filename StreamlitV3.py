import streamlit as st
import pandas as pd
import joblib
import pickle
import requests
import matplotlib.pyplot as plt
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

# Define a class to interact with the SimFin API
class PySimFin:
    def __init__(self, api_key):
        self.base_url = "https://backend.simfin.com/api/v3"
        self.headers = {'Authorization': api_key}
        logging.info("PySimFin instance created with API key.")

    def get_share_prices(self, ticker, start, end):
        url = f"{self.base_url}/companies/prices/compact?ticker={ticker}&start={start}&end={end}"
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()  # Will raise an HTTPError for bad requests (400+)
            data = response.json()
            df = pd.DataFrame(data[0]['data'])
            return df
        except requests.RequestException as e:
            logging.error(f"Request failed: {e}")
            return pd.DataFrame()

def simulate_trading(df):
    initial_balance = 10000  # Initial balance to simulate trading
    stocks_held = 0
    balance = initial_balance
    prices = df['Close']
    sma_14 = df['SMA_14']
    transactions = []  # List to store transaction records

    for i in range(len(df) - 1):  # loop through the data frame
        today_close = prices.iloc[i]
        tomorrow_close = prices.iloc[i+1]
        today_sma = sma_14.iloc[i]
        predicted_tomorrow = prediction[i]

        # Buy condition
        if balance != 0 and today_close < today_sma * 0.98 and predicted_tomorrow == 0:
            stocks_held = balance / tomorrow_close
            balance = 0  # all money is used to buy stocks
            transactions.append({'type': 'BUY', 'price': tomorrow_close, 'stocks': stocks_held})
            logging.info(f"Bought {stocks_held} stocks at {tomorrow_close} on {df.index[i + 1]}")
        
        # Sell condition
        if balance == 0 and today_close > today_sma * 1.02 and predicted_tomorrow == 1:
            balance = stocks_held * tomorrow_close
            transactions.append({'type': 'SELL', 'price': tomorrow_close, 'stocks': stocks_held})
            stocks_held = 0  # all stocks sold
            logging.info(f"Sold stocks at {tomorrow_close} on {df.index[i + 1]}, total balance: {balance}")

    return balance, stocks_held, transactions


def current_wallent_value (balance, stocks, today_price):
    if balance == 0:
        value = stocks*today_price
    else:
        value =0
    return value
ticker_mapping = {
    'AAPL': 1,
    'META': 2,
    'MSFT': 3,
    'NVDA': 4,
    'TSLA': 5,
    # Add more tickers as needed
}
# Load your model and scaler
model = joblib.load('DecisionTreeClassifier.pkl')
scaler = pickle.load(open("scaler_DecisionTree.pkl", "rb"))

# Initialize the PySimFin API wrapper
simfin_api = PySimFin(api_key='fde00d2f-38ad-43f3-9a83-012077d42da6')

# Streamlit app setup
st.sidebar.title("Navigation 🧭")
page = st.sidebar.radio("Select a Page:", ["Overview 📄", "Predict Next Day 🔮", "Show Financials 📊"])

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

# Load the CSV file directly into a pandas DataFrame
company_data = pd.read_csv('us-companies.csv', sep=";")
BS = pd.read_csv('us-balance-quarterly.csv', sep=";")
BS['YearQuarter'] = BS['Fiscal Year'].astype(str) + BS['Fiscal Period']

if page == "Overview 📄":
    st.title("Project Overview 🌟")
    st.write("""
        ### Project Description 📘
        This project focuses on predicting the next day's stock prices by leveraging historical data. Our approach combines advanced machine learning techniques with robust data processing to ensure accurate predictions.

        ### Tools and Resources 🔧
        - **Python**: Utilized for comprehensive data processing and machine learning tasks.
        - **GitHub**: Utilized as the central repository for all code, facilitating seamless collaboration and version control throughout the project.
        - **Streamlit**: Employed to develop an interactive web application that showcases our predictive models.
        - **ChatGPT**: Used as a coding assistant to help debug and optimize our codebase.
             
        ### Steps to Complete the Project 📈
        - **Stock Market Research**:
        - Conduct thorough research on stock market trends and factors influencing stock prices. Analyze historical data to identify patterns and potential predictors of stock movements.

        - **Data Extraction - Batch Data**:
        - Extract historical stock data from financial databases and files. This was used to download three main files companies details, share price, and balance sheet details.

        - **Data Extraction - API**:
        - Implement real-time data extraction via financial API. This was used to automate the predictions based on user selection in the final web page.

        - **Data Cleaning**:
        - Clean the extracted data by handling missing values, adding new columns based on business need, and ensuring data consistency. This step is vital to prepare the data for accurate and efficient model training.

        - **ML Model Development**:
        - Design and train machine learning models using the cleaned data. Explore various algorithms like Decision Trees, Random Forest, and Gradient Boosting to find the best predictor for stock price movements.

        - **Testing the model with the API**:
        - Integrate the trained model with the stock data API to test its performance in a real-world scenario.

        - **User-experience Web Page Development**:
        - Develop a user-friendly web interface using Streamlit. This interface should allow users to select stocks, view predictions, and receive updates on stock movements. Ensure the page is intuitive and provides a seamless experience with visual representations of data and predictions.


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
        df = simfin_api.get_share_prices(ticker, "2025-01-01", "2025-06-01")
        if not df.empty:
            df = prepare_data(df)
            df['Ticker_cat'] = ticker_mapping[ticker]
            plt.figure(figsize=(10, 4))
            plt.plot(pd.to_datetime(df['Date'][:90]), df['Close'], label='Close Price')
            plt.plot(pd.to_datetime(df['Date'][:90]), df['SMA_14'], label='14-Day SMA', color='red')
            plt.title('Close Price by Day for the Last Year')
            plt.xlabel('Date')
            plt.ylabel('Close Price ($)')
            plt.legend()
            st.pyplot(plt)

            
            columns_to_scale = ['Ticker_cat','Open', 'High', 'Low', 'Close', 'Adj. Close', 'Volume', 'week', 'First_day', 'Last_day', 'SMA_7', 'V_SMA_7', 'SMA_14', 'V_SMA_14', 'EMA_7', 'V_EMA_7', 'EMA_14', 'V_EMA_14']
            X_scaled = scaler.transform(df[columns_to_scale])
            prediction = model.predict(X_scaled)
            st.write("Prediction for next day:", "🟢 Positive 🟢" if prediction[0] else "🟠 Negative 🟠")

            balance,stocks_held,transactions = simulate_trading(df)
            current_wallet = current_wallent_value(balance, stocks_held,df['Close'].iloc[-1])
            st.info(f"The applied strategy is: if tomorrow predicted price is negative and today closing price is 2% less than SMA14 days then buy with all existing budget, and only sell when opoosite happen ")
            st.title(f"Summary of investments in {ticker}:")
            st.markdown(f"**Starting Balance:** 10,000 **| Current Balance:** {balance:,.2f} **| Number of Stocks:** {stocks_held}")
            st.markdown(f'**Current balance as of today price:** {current_wallet:.1f}","🟢" if current_wallet<balance else "🟠"')
            for transaction in transactions:
                st.write(transaction)

elif page == "Show Financials 📊":
    st.title("Financial Overview 📈")
    ticker = st.selectbox("Select Ticker for Financials", ['TSLA', 'AAPL', 'MSFT', 'NVDA', 'META'])
    company_financials = company_data[company_data['Ticker'] == ticker]
    company_financials_2 = company_financials[[ 'Company Name', 'Number Employees', 
       'Market', 'Main Currency']]
    BusinessSummary = company_financials['Business Summary']
    st.dataframe(company_financials_2)
    st.title("Company Overview")
    st.info(BusinessSummary)

    BS = BS[BS['Ticker'] == ticker]
    plt.figure(figsize=(10, 4))
    plt.bar(BS['YearQuarter'][:12], BS['Total Assets'][:12], label='Total Assetes by Quarter')
    plt.title('Total Assetes by Quarter')
    plt.xlabel('YearQuarter')
    plt.ylabel('Total Assetes')
    plt.legend()
    st.pyplot(plt)

    BS = BS[BS['Ticker'] == ticker]
    plt.figure(figsize=(10, 4))
    plt.bar(BS['YearQuarter'][:12], BS['Total Liabilities'][:12], label='Total Liabilities by Quarter')
    plt.title('Total Liabilities by Quarter')
    plt.xlabel('YearQuarter')
    plt.ylabel('Total Liabilities')
    plt.legend()
    st.pyplot(plt)

    BS = BS[BS['Ticker'] == ticker]
    plt.figure(figsize=(10, 4))
    plt.bar(BS['YearQuarter'][:12], BS['Total Equity'][:12], label='Total Equity by Quarter')
    plt.title('Total Equity by Quarter')
    plt.xlabel('YearQuarter')
    plt.ylabel('Total Equity')
    plt.legend()
    st.pyplot(plt)