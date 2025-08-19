import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import yfinance as yf
import streamlit as st
import plotly.express as px


def create_lag_features(df, target_col="Close", n_lags=5):
    """
    Create lag features for time series forecasting.
    """
    for lag in range(1, n_lags + 1):
        df[f"lag_{lag}"] = df[target_col].shift(lag)
    return df.dropna()


def run_xgboost_model(df, target_col="Close", n_lags=5, train_ratio=0.8):
    """
    Train XGBoost model on lag features.
    """
    # Create lag features
    df = create_lag_features(df, target_col, n_lags)

    # Split train/test
    train_size = int(len(df) * train_ratio)
    train_df, test_df = df.iloc[:train_size], df.iloc[train_size:]

    X_train, y_train = train_df.drop(columns=[target_col]), train_df[target_col]
    X_test, y_test = test_df.drop(columns=[target_col]), test_df[target_col]

    # Train model
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)

    # Predictions
    test_df = X_test.copy()
    test_df["XGB_Forecast"] = model.predict(X_test)
    test_df[target_col] = y_test

    # Metrics
    rmse = np.sqrt(mean_squared_error(test_df[target_col], test_df["XGB_Forecast"]))
    mae = mean_absolute_error(test_df[target_col], test_df["XGB_Forecast"])
    mape = np.mean(np.abs((test_df[target_col] - test_df["XGB_Forecast"]) / test_df[target_col])) * 100

    metrics = {"RMSE": rmse, "MAE": mae, "MAPE": mape}

    return model, test_df, metrics


def plot_forecast(test_df, target_col="Close"):
    """
    Plot actual vs forecasted values.
    """
    fig = px.line(test_df[[target_col, "XGB_Forecast"]], 
                  labels={"value": "Price", "index": "Time"}, 
                  title="XGBoost Time Series Forecast")
    return fig
