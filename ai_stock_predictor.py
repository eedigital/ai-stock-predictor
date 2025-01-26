import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import yfinance as yf
import matplotlib.pyplot as plt

class MarketPredictor:
    def __init__(self, symbols, start_date, end_date):
        self.symbols = symbols  # List of symbols for stocks and commodities
        self.start_date = start_date
        self.end_date = end_date
        self.data = {}
        self.models = {}

    def fetch_data(self):
        """Fetch historical data for all symbols."""
        for symbol in self.symbols:
            try:
                print(f"Fetching data for {symbol}...")
                df = yf.download(symbol, start=self.start_date, end=self.end_date)
                if df.empty:
                    print(f"No data found for {symbol}. Skipping...")
                    continue
                df['Change'] = df['Close'].pct_change()
                df['SMA_20'] = df['Close'].rolling(window=20).mean()  # Simple Moving Average
                df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()  # Exponential Moving Average
                df['Volatility'] = df['Close'].rolling(window=20).std()  # Volatility indicator
                df.dropna(inplace=True)
                self.data[symbol] = df
            except Exception as e:
                print(f"Error fetching data for {symbol}: {e}")

    def prepare_features(self, symbol):
        """Prepare features and target for training and testing for a specific symbol."""
        df = self.data[symbol]
        df['Lag1'] = df['Change'].shift(1)
        df['Lag2'] = df['Change'].shift(2)
        df.dropna(inplace=True)

        X = df[['Lag1', 'Lag2', 'SMA_20', 'EMA_20', 'Volatility']]
        y = df['Change']
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def train_models(self):
        """Train predictive models for all symbols."""
        for symbol in self.symbols:
            if symbol not in self.data:
                print(f"No data available for {symbol}. Skipping model training.")
                continue
            print(f"Training model for {symbol}...")
            X_train, X_test, y_train, y_test = self.prepare_features(symbol)
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            mse = mean_squared_error(y_test, predictions)
            print(f"Model for {symbol} trained. Mean Squared Error: {mse}")
            self.models[symbol] = model

    def predict_future(self, symbol, days_ahead):
        """Predict future changes for a specified number of days."""
        if symbol in self.data and symbol in self.models:
            future_predictions = []
            last_data = self.data[symbol][['Lag1', 'Lag2', 'SMA_20', 'EMA_20', 'Volatility']].iloc[-1].values

            for _ in range(days_ahead):
                prediction = self.models[symbol].predict([last_data])[0]
                future_predictions.append(prediction * 100)

                # Update last_data with the predicted change (Lag1 and Lag2 shift forward)
                last_data[0] = prediction  # Update Lag1 with the predicted change
                last_data[1] = last_data[0]  # Update Lag2

            return future_predictions
        else:
            print(f"No data or model available for {symbol}.")
            return None

# Example Usage
if __name__ == "__main__":
    symbols = ["AAPL", "GOOGL", "CL=F", "CORN", "XLE"]  # Technology, Alphabet, Crude Oil, Corn, Energy ETFs
    start_date = "2020-01-01"
    end_date = "2023-01-01"

    predictor = MarketPredictor(symbols, start_date, end_date)
    predictor.fetch_data()
    predictor.train_models()

    for symbol in symbols:
        print(f"Predicted changes for {symbol} in 2025:")
        predictions = predictor.predict_future(symbol, days_ahead=5)  # Predict for the next 5 days
        if predictions:
            print(predictions)

    # Visualization
    for symbol in symbols:
        if symbol not in predictor.data:
            print(f"No data available for {symbol}. Skipping plot.")
            continue
        df = predictor.data[symbol]
        plt.figure(figsize=(10, 6))
        plt.plot(df.index, df['Close'], label="Actual Close Prices")
        plt.title(f"{symbol} Close Price Over Time")
        plt.legend()
        plt.show()
