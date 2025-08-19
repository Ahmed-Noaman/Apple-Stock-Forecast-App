import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
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


def run_random_forest_model(df, target_col="Close", n_lags=5, train_ratio=0.8):
    """
    Train Random Forest model on lag features.
    """
    # Create lag features
    df = create_lag_features(df, target_col, n_lags)

    # Split train/test
    train_size = int(len(df) * train_ratio)
    train_df, test_df = df.iloc[:train_size], df.iloc[train_size:]

    X_train, y_train = train_df.drop(columns=[target_col]), train_df[target_col]
    X_test, y_test = test_df.drop(columns=[target_col]), test_df[target_col]

    # Train model
    model = RandomForestRegressor(
        n_estimators=200, 
        max_depth=10, 
        random_state=42, 
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # Predictions
    test_df = X_test.copy()
    test_df["RF_Forecast"] = model.predict(X_test)
    test_df[target_col] = y_test

    # Metrics
    rmse = np.sqrt(mean_squared_error(test_df[target_col], test_df["RF_Forecast"]))
    mae = mean_absolute_error(test_df[target_col], test_df["RF_Forecast"])
    mape = np.mean(np.abs((test_df[target_col] - test_df["RF_Forecast"]) / test_df[target_col])) * 100

    metrics = {"RMSE": rmse, "MAE": mae, "MAPE": mape}

    return model, test_df, metrics


def plot_forecast(test_df, target_col="Close"):
    """
    Plot actual vs forecasted values.
    """
    fig = px.line(
        test_df[[target_col, "RF_Forecast"]], 
        labels={"value": "Price", "index": "Time"}, 
        title="Random Forest Time Series Forecast"
    )
    return fig


# # ========== Streamlit App ========== #
# def run_app():
#     st.title("🌲 Random Forest Time Series Forecasting")

#     ticker = st.text_input("Enter Stock Ticker:", "AAPL")
#     period = st.selectbox("Select Period", ["1y", "2y", "5y"], index=0)
#     n_lags = st.slider("Number of Lag Features", 1, 20, 5)

#     if st.button("Run Forecast"):
#         # Download Data
#         df = yf.download(ticker, period=period)
#         df = df[["Close"]]

#         # Train model
#         model, test_df, metrics = run_random_forest_model(df, target_col="Close", n_lags=n_lags)

#         # Show Metrics
#         st.subheader("📊 Model Evaluation Metrics")
#         st.write(f"**RMSE:** {metrics['RMSE']:.4f}")
#         st.write(f"**MAE:** {metrics['MAE']:.4f}")
#         st.write(f"**MAPE:** {metrics['MAPE']:.2f}%")

#         # Plot Forecast
#         fig = plot_forecast(test_df, target_col="Close")
#         st.plotly_chart(fig)


# if __name__ == "__main__":
#     run_app()
