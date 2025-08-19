from statsmodels.tsa.api import VAR
import pandas as pd

def run_var_model(data, lags=1):
    """
    VAR model forecast
    Requires multiple time series columns (e.g., Close, Volume, etc.)
    """
    # Fit the model
    model = VAR(data)
    model_fit = model.fit(lags)

    # Get the last 'lags' observations for forecasting
    last_obs = data.values[-lags:]

    # Forecast next step
    prediction = model_fit.forecast(last_obs, steps=1)

    # Create forecast DataFrame
    forecast_df = pd.DataFrame(prediction, columns=[col + "_VAR_Forecast" for col in data.columns])

    # Append forecast row to original data
    result_df = pd.concat([data, forecast_df], ignore_index=True)

    return result_df
