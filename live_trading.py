import alpaca_trade_api as tradeapi
import joblib
import pandas as pd
import numpy as np
import pandas_ta as ta
from sklearn.preprocessing import StandardScaler
import time

# Load model and scaler
model = joblib.load("mean_reversion_model.pkl")
scaler = joblib.load("scaler.pkl")

# Alpaca API credentials
API_KEY = "INSERT UR KEY HERE"
SECRET_KEY = "INSERT UR KEY HERE"
BASE_URL = "https://paper-api.alpaca.markets"

# Initialize Alpaca API
api = tradeapi.REST(API_KEY, SECRET_KEY, BASE_URL, api_version="v2")

# Model input features
features = [
    "Price_Change_Pct", "BB_UPPER", "BB_MIDDLE", "BB_LOWER",
    "RSI_14", "ATR_14", "SMA_20", "EMA_20"
]

def fetch_and_preprocess_data(symbol):
    try:
        print(f"Fetching market data for {symbol}...")
        bars = api.get_bars(symbol, "1Min", limit=20).df
        bars = bars[bars["symbol"] == symbol]

        if bars.empty or not all(col in bars.columns for col in ["open", "high", "low", "close", "volume"]):
            print("Insufficient data. Skipping.")
            return pd.DataFrame()

        bars["Price_Change_Pct"] = bars["close"].pct_change()
        bb = ta.bbands(bars["close"], length=20)
        bars["BB_UPPER"], bars["BB_MIDDLE"], bars["BB_LOWER"] = bb["BBU_20_2.0"], bb["BBM_20_2.0"], bb["BBL_20_2.0"]
        bars["RSI_14"] = ta.rsi(bars["close"], length=14)
        bars["ATR_14"] = ta.atr(bars["high"], bars["low"], bars["close"], length=14)
        bars["SMA_20"] = ta.sma(bars["close"], length=20)
        bars["EMA_20"] = ta.ema(bars["close"], length=20)

        bars.dropna(inplace=True)
        if len(bars) == 0:
            print("No valid rows after feature processing.")
            return pd.DataFrame()

        data_scaled = scaler.transform(bars[features])
        bars["Prediction"] = model.predict_proba(data_scaled)[:, 1]

        print("Latest prediction:", bars["Prediction"].iloc[-1])
        return bars
    except Exception as e:
        print(f"Data fetch error: {e}")
        return pd.DataFrame()

def already_holding(symbol):
    try:
        position = api.get_position(symbol)
        return float(position.qty) > 0
    except:
        return False  # No position exists

def open_order_exists(symbol):
    try:
        orders = api.list_orders(status='open')
        return any(order.symbol == symbol for order in orders)
    except Exception as e:
        print(f"Order check error: {e}")
        return False

def execute_trade(symbol):
    data = fetch_and_preprocess_data(symbol)
    if data.empty:
        return

    latest_pred = data["Prediction"].iloc[-1]
    current_price = data["close"].iloc[-1]
    print(f"Prediction: {latest_pred:.4f} | Price: {current_price:.2f}")

    if latest_pred > 0.7:
        if already_holding(symbol):
            print(f"Already holding {symbol}. Skipping buy.")
            return
        if open_order_exists(symbol):
            print(f"Open order for {symbol} already exists. Skipping buy.")
            return

        try:
            cash = float(api.get_account().cash)
            quantity = int(cash // current_price)

            if quantity == 0:
                print("Not enough cash to buy.")
                return

            # Buy market order
            api.submit_order(
                symbol=symbol,
                qty=quantity,
                side="buy",
                type="market",
                time_in_force="gtc"
            )
            print(f"BUY {quantity} shares of {symbol} at ${current_price:.2f}")

            # Simulate OCO with separate stop and limit
            stop_loss = round(current_price * 0.97, 2)
            take_profit = round(current_price * 1.05, 2)

            api.submit_order(
                symbol=symbol,
                qty=quantity,
                side="sell",
                type="stop",
                time_in_force="gtc",
                stop_price=stop_loss
            )

            api.submit_order(
                symbol=symbol,
                qty=quantity,
                side="sell",
                type="limit",
                time_in_force="gtc",
                limit_price=take_profit
            )

            print(f"Stop-loss at ${stop_loss}, Take-profit at ${take_profit}")

        except Exception as e:
            print(f"Trade execution error: {e}")
    else:
        print("Prediction below threshold. No trade.")

def main():
    symbol = "AAPL"
    while True:
        try:
            execute_trade(symbol)
        except Exception as e:
            print(f"Main loop error: {e}")
        time.sleep(30)

if __name__ == "__main__":
    main()
