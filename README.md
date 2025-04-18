# 📉 Meany Bot 
### Mean Reversion Trading Bot

This project implements a fully functional **algorithmic trading bot** that leverages **mean reversion strategies** powered by **machine learning**, real-time price monitoring, and backtesting infrastructure. The system is built using Python, integrates with the Alpaca API for live execution, and applies technical indicators to detect short-term price reversals in equities.

---

## 🧠 Strategy Overview

Mean reversion strategies exploit the tendency of stock prices to return to their historical average after significant short-term deviations. This bot identifies these conditions using technical indicators and classifies whether a price reversion is likely to occur.

---

## 📂 Project Structure

```
mean-reversion-trading-bot/
├── backtesting.py                   # Backtesting engine using historical data
├── features.py                      # Feature engineering and technical indicators
├── historical_stock_data.csv        # Raw price data (Yahoo Finance)
├── live_trading.py                  # Real-time trading bot using Alpaca API
├── mean_reversion_model.pkl         # Trained Random Forest model
├── model_training.py                # ML model training and evaluation pipeline
├── scaler.pkl                       # Scaler used for feature normalization
├── stock_data_with_indicators.csv   # Processed data with features
└── README.md                        # Project documentation
```

---

## ⚙️ Components

### 🔍 1. Feature Engineering (`features.py`)
- Calculates:
  - **Bollinger Bands** (volatility-based overbought/oversold detection)
  - **RSI** (momentum oscillator)
  - **ATR** (volatility for dynamic stop-loss)
  - **% Price Change** (price deviation over a time window)

### 🧠 2. Model Training (`model_training.py`)
- Preprocesses data and extracts features
- Trains a **Random Forest Classifier** to predict mean reversion
- Outputs:
  - `mean_reversion_model.pkl` – model used during backtesting and live trading
  - `scaler.pkl` – normalization parameters for live predictions

### 💹 3. Backtesting (`backtesting.py`)
- Simulates trades based on predictions from the trained model
- Tracks:
  - Annual ROI
  - Sharpe ratio
  - Win/loss rate
- Helps evaluate performance and optimize strategy before deployment

### ⚡ 4. Live Trading (`live_trading.py`)
- Connects to the **Alpaca Paper Trading API**
- Runs inference using live market data and executes trades automatically
- Includes risk management:
  - Position sizing
  - Stop-loss / take-profit logic
  - Trade cooldowns

---

## 🧪 How to Use

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Model
```bash
python model_training.py
```

### 3. Run Backtests
```bash
python backtesting.py
```

### 4. Start Live Trading
Make sure to set your Alpaca API keys as environment variables or in a `.env` file:
```bash
export APCA_API_KEY_ID='your_key'
export APCA_API_SECRET_KEY='your_secret'
python live_trading.py
```

---

## 🧑‍💻 Author

Made with <3 by **Manroop Kalsi**  
