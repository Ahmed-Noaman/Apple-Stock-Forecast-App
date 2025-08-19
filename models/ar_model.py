
from statsmodels.tsa.ar_model import AutoReg

def run_ar_model(data):
    model = AutoReg(data['Close'], lags=1).fit()
    prediction = model.predict(start=len(data), end=len(data))
    data['AR Forecast'] = prediction
    return data
