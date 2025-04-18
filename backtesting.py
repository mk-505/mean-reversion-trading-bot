import backtrader as bt
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Load the trained model and scaler
model = joblib.load("mean_reversion_model.pkl")
scaler = joblib.load("scaler.pkl")

# Load the preprocessed data
data = pd.read_csv("stock_data_with_indicators.csv")

# Define features used in the model
features = [
    "Price_Change_Pct", "BB_UPPER", "BB_MIDDLE", "BB_LOWER",
    "RSI_14", "ATR_14", "SMA_20", "EMA_20"
]

# Add predictions to the data
data_scaled = scaler.transform(data[features])
data["Prediction"] = model.predict_proba(data_scaled)[:, 1]  # Probability of mean reversion

# Define a Backtrader strategy
class MeanReversionStrategy(bt.Strategy):
    params = (
        ("reversion_threshold", 0.5),  # Probability threshold to trigger a trade
        ("holding_period", 5),        # Holding period for each trade
    )

    def __init__(self):
        self.prediction = self.data.Prediction
        self.order = None
        self.bar_executed = None

    def next(self):
        if self.order:
            return  # Skip if an order is pending

        # Buy signal
        if self.prediction[0] > self.params.reversion_threshold:
            self.order = self.buy(size=1)
            self.bar_executed = len(self)

        # Sell after holding period
        if self.bar_executed is not None and len(self) >= (self.bar_executed + self.params.holding_period):
            self.order = self.sell(size=1)

    def notify_order(self, order):
        if order.status in [order.Completed, order.Canceled, order.Margin]:
            self.order = None

# Prepare data for Backtrader
data["Date"] = pd.to_datetime(data["Date"])
data.set_index("Date", inplace=True)

# Custom data feed with Prediction column
class CustomPandasData(bt.feeds.PandasData):
    lines = ('Prediction',)
    params = (('Prediction', -1),)

# Create the Backtrader data feed
data_bt = CustomPandasData(dataname=data)

# Initialize Cerebro engine
cerebro = bt.Cerebro()
cerebro.addstrategy(MeanReversionStrategy)
cerebro.adddata(data_bt)
cerebro.broker.set_cash(10000.0)

# Add analyzers
cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe")
cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")
cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")

# Run the backtest
print("Starting Portfolio Value: %.2f" % cerebro.broker.getvalue())
results = cerebro.run()
print("Final Portfolio Value: %.2f" % cerebro.broker.getvalue())

# Extract analyzers
strat = results[0]
sharpe_ratio = strat.analyzers.sharpe.get_analysis().get("sharperatio", None)
drawdown_data = strat.analyzers.drawdown.get_analysis()
returns_data = strat.analyzers.returns.get_analysis()
trade_analysis = strat.analyzers.trades.get_analysis()

# Extract key performance metrics
total_return = returns_data.get("rtot", None)
annual_roi = returns_data.get("rnorm", None)

# Win rate
try:
    win_rate = trade_analysis.won.total / trade_analysis.total.closed * 100
    print(f"Win Rate: {win_rate:.2f}%")
except Exception as e:
    print(f"Could not calculate win rate: {e}")
    win_rate = None

# Output metrics
print(f"Sharpe Ratio: {sharpe_ratio:.2f}" if sharpe_ratio is not None else "Sharpe Ratio: N/A")
print(f"Max Drawdown: {drawdown_data['max']['drawdown']:.2f}%" if 'max' in drawdown_data else "Max Drawdown: N/A")
print(f"Total Return: {total_return * 100:.2f}%" if total_return is not None else "Total Return: N/A")
print(f"Annual ROI: {annual_roi * 100:.2f}%" if annual_roi is not None else "Annual ROI: N/A")
