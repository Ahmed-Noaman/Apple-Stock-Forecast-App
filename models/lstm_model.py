
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error, mean_absolute_error
import yfinance as yf

# ------------------------------
# LSTM Module
# ------------------------------
class LSTMModel:
    def __init__(self, look_back=10, epochs=50, batch_size=32):
        self.look_back = look_back
        self.epochs = epochs
        self.batch_size = batch_size
        self.scaler = MinMaxScaler()
        self.model = None

    def create_sequences(self, data):
        """ تحويل البيانات إلى تسلسلات مناسبة للـ LSTM """
        X, y = [], []
        for i in range(len(data) - self.look_back):
            X.append(data[i:(i + self.look_back)])
            y.append(data[i + self.look_back])
        return np.array(X), np.array(y)

    def fit(self, df, target_col="Close"):
        """ تدريب الموديل على البيانات """
        values = df[target_col].values.reshape(-1, 1)
        scaled = self.scaler.fit_transform(values)

        X, y = self.create_sequences(scaled)

        # reshape [samples, timesteps, features]
        X = X.reshape((X.shape[0], X.shape[1], 1))

        # بناء الموديل
        self.model = Sequential()
        self.model.add(LSTM(50, return_sequences=True, input_shape=(self.look_back, 1)))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(50))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(1))

        self.model.compile(optimizer="adam", loss="mse")

        self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=1)

    def predict(self, df, target_col="Close"):
        """ التنبؤ بالقيم القادمة """
        values = df[target_col].values.reshape(-1, 1)
        scaled = self.scaler.transform(values)

        X, y = self.create_sequences(scaled)
        X = X.reshape((X.shape[0], X.shape[1], 1))

        predictions = self.model.predict(X)
        predictions = self.scaler.inverse_transform(predictions)

        # حساب المقاييس
        y_true = self.scaler.inverse_transform(y.reshape(-1, 1))

        rmse = np.sqrt(mean_squared_error(y_true, predictions))
        mae = mean_absolute_error(y_true, predictions)
        mape = np.mean(np.abs((y_true - predictions) / y_true)) * 100

        metrics = {
            "RMSE": rmse,
            "MAE": mae,
            "MAPE": mape
        }

        # تجهيز test dataset
        test_df = df.iloc[self.look_back:].copy()
        test_df["LSTM_Prediction"] = predictions

        return test_df, metrics



# ------------------------------
# دالة التشغيل الرئيسية لاستخدامها في Streamlit
# ------------------------------
def run_LSTM_model(df):
    lstm = LSTMModel(look_back=20, epochs=30, batch_size=16)
    lstm.fit(df, target_col="Close")
    test_df, metrics = lstm.predict(df, target_col="Close")
    return test_df, metrics

