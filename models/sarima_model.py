from statsmodels.tsa.statespace.sarimax import SARIMAX

def run_sarima_model(data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)):
    """
    SARIMA model forecast
    order: (p, d, q)
    seasonal_order: (P, D, Q, s) where s is seasonal period
    """
    model = SARIMAX(data['Close'], order=order, seasonal_order=seasonal_order)
    model_fit = model.fit(disp=False)
    prediction = model_fit.predict(start=len(data), end=len(data))
    data["SARIMA Forecast"] = prediction
    return data
