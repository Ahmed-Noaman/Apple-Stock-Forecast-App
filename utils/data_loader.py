
import yfinance as yf
import pandas as pd
import streamlit as st

@st.cache_data
def load_data():
    df = yf.download("AAPL", start="2010-01-01", end="2024-12-31", auto_adjust=True)
    df = df[["Close", "Open", "High", "Low"]].dropna()
    df.columns = ["Close", "Open", "High", "Low"]
    return df
