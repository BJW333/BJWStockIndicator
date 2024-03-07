import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from alpha_vantage.timeseries import TimeSeries
import warnings
warnings.filterwarnings("ignore")
import datetime
current_time = datetime.datetime.now()
formatted_time = current_time.strftime('%I:%M:%S')
   
# Add your Alpha Vantage API Key here
API_KEY = 'MJX8BVSA9W1WOEH4'

# Define the stocks to monitor
stocks = ['GOOGL', 'MSFT', 'AMZN', 'TSLA', 'NVDA', 'AAPL', 'V', 'META', 'JPM', 'ORCL', 'HD', 'XOM', 'TSM']

# Function to compute Relative Strength Index (RSI)
def compute_rsi(data, window):
    diff = data.diff(1)
    up_chg = 0 * diff
    down_chg = 0 * diff
    up_chg[diff > 0] = diff[diff > 0]
    down_chg[diff < 0] = diff[diff < 0]
    up_chg_avg = up_chg.ewm(com=window - 1, min_periods=window).mean()
    down_chg_avg = down_chg.ewm(com=window - 1, min_periods=window).mean()
    rs = up_chg_avg / down_chg_avg
    rsi = 100 - 100 / (1 + rs)
    return rsi

# Function to fetch real-time data
def fetch_realtime_data(symbol):
    ts = TimeSeries(key=API_KEY, output_format='pandas')
    #change interval to once a day
    data, _ = ts.get_intraday(symbol=symbol, interval='1min', outputsize='full')
    return data

# Function to train the Random Forest model
def train_model(X, Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X_train, Y_train)
    return model

# Loop to continuously fetch data and generate signals
while True:
    for stock in stocks:
        # Fetch real-time data
        data = fetch_realtime_data(stock)

        # Compute technical indicators
        data['RSI'] = compute_rsi(data['4. close'], 14)
        data.dropna(inplace=True)

        # Prepare data for model training
        X = data[['RSI']]
        y = data['4. close'].shift(-1)
        y.drop(y.tail(1).index, inplace=True) 
        X.drop(X.tail(1).index, inplace=True) 

        # Train the model
        model = train_model(X, y)

        # Use the model to predict the next minute price
        X_latest = data[['RSI']].tail(1)
        prediction = model.predict(X_latest)

        # Generate trading signal
        if prediction > data['4. close'].iloc[-1] * 1.01:
            print(f"Buy signal for {stock}")
            print(formatted_time)
        elif prediction < data['4. close'].iloc[-1] * 0.99:
            print(f"Sell signal for {stock}")
            print(formatted_time)
        else:
            print(f"Hold signal for {stock}")
            print(formatted_time)

        # Sleep for a minute before fetching data for the next stock
        time.sleep(30)
        #time.sleep(60) normal amount
