
def run_naive_forecast(data):
    data["naive_forecast"] = data["Close"].shift(1)
    return data
