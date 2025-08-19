from statsmodels.tsa.arima.model import ARIMA

def run_arima_model(data, order=(5, 1, 0)):
    """
    ARIMA model forecast
    order: (p, d, q)
    """
    model = ARIMA(data['Close'], order=order)
    model_fit = model.fit()
    prediction = model_fit.predict(start=len(data), end=len(data))
    data["ARIMA Forecast"] = prediction
    return data
