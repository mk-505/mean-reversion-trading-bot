import yfinance as yf
import pandas as pd
import ta

# Define the stock tickers and time period
tickers = ["AAPL", "TSLA", "SPY"]
start_date = "2018-01-01"
end_date = "2023-01-01"

# Download historical data
data = yf.download(tickers, start=start_date, end=end_date, group_by='ticker')

# Save raw data to a CSV file (optional)
data.to_csv("historical_stock_data.csv")

# Function to calculate technical indicators for a single stock
def calculate_indicators(df):
    df = df.copy()  # Ensure we don't modify the original DataFrame

    # Check if necessary columns exist to avoid errors
    if "Close" not in df or "High" not in df or "Low" not in df:
        print("Missing required columns")
        return df

    # Bollinger Bands
    df["BB_UPPER"] = ta.volatility.bollinger_hband(df["Close"], window=20)
    df["BB_MIDDLE"] = ta.volatility.bollinger_mavg(df["Close"], window=20)
    df["BB_LOWER"] = ta.volatility.bollinger_lband(df["Close"], window=20)

    # RSI (Relative Strength Index)
    df["RSI_14"] = ta.momentum.rsi(df["Close"], window=14)

    # ATR (Average True Range)
    df["ATR_14"] = ta.volatility.average_true_range(df["High"], df["Low"], df["Close"], window=14)

    # Moving Averages (SMA and EMA)
    df["SMA_20"] = ta.trend.sma_indicator(df["Close"], window=20)
    df["EMA_20"] = ta.trend.ema_indicator(df["Close"], window=20)

    # Price Change %
    df["Price_Change_Pct"] = df["Close"].pct_change() * 100

    return df.dropna()  # Drop NaN rows due to indicator calculations

# Store each processed DataFrame separately
processed_data = {}

for ticker in tickers:
    processed_data[ticker] = calculate_indicators(data[ticker])

# Combine processed data for all tickers into a single DataFrame
all_data = pd.concat(processed_data, names=["Ticker", "Date"]).reset_index()

# Save the final dataset to a CSV file (optional)
all_data.to_csv("stock_data_with_indicators.csv", index=False)

# Display the final dataset
print(all_data.head())
