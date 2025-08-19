import pandas as pd

def run_ma_model(data, window=3):
    """
    Moving Average Forecast
    window: number of past observations to average
    """
    data["MA Forecast"] = data["Close"].rolling(window=window).mean().shift(1)
    return data
