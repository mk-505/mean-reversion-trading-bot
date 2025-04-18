import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, roc_auc_score

# Load the preprocessed data from features.py
data = pd.read_csv("stock_data_with_indicators.csv")

# Define the target variable: Mean Reversion (1) or No Reversion (0)
# Here, we assume mean reversion occurs if the price reverts to the mean within a certain window
# For simplicity, let's define mean reversion as:
# If the price increases by > 1% within the next 5 days after a drop, label it as 1 (reversion)
# Otherwise, label it as 0 (no reversion)

# Create the target variable
data["Target"] = np.where(
    data.groupby("Ticker")["Close"].shift(-5) / data["Close"] - 1 > 0.01, 1, 0
)

# Drop rows with NaN values (due to shifting)
data.dropna(inplace=True)

# Features and target
features = [
    "Price_Change_Pct", "BB_UPPER", "BB_MIDDLE", "BB_LOWER", "RSI_14", "ATR_14", "SMA_20", "EMA_20"
]
X = data[features]
y = data["Target"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features (important for Logistic Regression and XGBoost)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

import joblib
joblib.dump(scaler, "scaler.pkl")

# Function to evaluate model performance
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability of class 1

    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print(f"Precision: {precision_score(y_test, y_pred):.2f}")
    print(f"Recall: {recall_score(y_test, y_pred):.2f}")
    print(f"ROC AUC: {roc_auc_score(y_test, y_pred_proba):.2f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

# Model: Random Forest
print("Training Random Forest...")
rf = RandomForestClassifier(random_state=42, n_estimators=100)
rf.fit(X_train_scaled, y_train)
print("Random Forest Performance:")
evaluate_model(rf, X_test_scaled, y_test)

# Save the best model (Random Forest) for future use
joblib.dump(rf, "mean_reversion_model.pkl")


# Example: Predict probability of mean reversion for a new data point
new_data = np.array([[0.5, 150, 145, 140, 55, 2.5, 148, 147]])  # Example feature values
new_data_scaled = scaler.transform(new_data)
probability = rf.predict_proba(new_data_scaled)[0][1]
print(f"Probability of Mean Reversion: {probability * 100:.2f}%")
