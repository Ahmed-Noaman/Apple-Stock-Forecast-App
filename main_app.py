
import streamlit as st
import streamlit.components.v1 as components
from utils.data_loader import load_data
from models.ar_model import run_ar_model
from models.ma_model import run_ma_model
from models.arima_model import run_arima_model
from models.sarima_model import run_sarima_model
from models.var_model import run_var_model
from models.naive_model import run_naive_forecast
from models.xgboost_model import run_xgboost_model
from models.random_forest_model import run_random_forest_model
from models.lstm_model import run_LSTM_model




data = load_data()

st.title("📈 Apple Stock Forecasting App")
model_option = st.sidebar.radio("Model Type:", ["Naive Forecasting", "AR", "MA", "ARIMA", "SARIMA", "VAR","XGBoost","Random Forest","LSTM"])

def load_html(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

if model_option == "Naive Forecasting":
    tab1, tab2 = st.tabs(["🔮 Prediction", "📘 Method Explanation"])
    with tab1:
        df_forecast = run_naive_forecast(data)
        st.line_chart(df_forecast.tail(100))
    with tab2:
        components.html(load_html("explanations_html/naive.html"), height=600, scrolling=True)

elif model_option == "AR":
    tab1, tab2 = st.tabs(["🔮 Prediction", "📘 Method Explanation"])
    with tab1:
        df_forecast = run_ar_model(data)
        st.line_chart(df_forecast.tail(100))
    with tab2:
        components.html(load_html("explanations_html/ar.html"), height=600, scrolling=True)
        
elif model_option == "MA":
    tab1, tab2 = st.tabs(["🔮 Prediction", "📘 Method Explanation"])
    with tab1:
        df_forecast = run_ma_model(data)
        st.line_chart(df_forecast.tail(100))
    with tab2:
        components.html(load_html("explanations_html/ma.html"), height=600, scrolling=True)


elif model_option == "ARIMA":
    tab1, tab2 = st.tabs(["🔮 Prediction", "📘 Method Explanation"])
    with tab1:
        df_forecast = run_arima_model(data)
        st.line_chart(df_forecast.tail(100))
    with tab2:
        components.html(load_html("explanations_html/arima.html"), height=600, scrolling=True)

elif model_option == "SARIMA":
    tab1, tab2 = st.tabs(["🔮 Prediction", "📘 Method Explanation"])
    with tab1:
        df_forecast = run_sarima_model(data)
        st.line_chart(df_forecast.tail(100))
    with tab2:
        components.html(load_html("explanations_html/sarima.html"), height=600, scrolling=True)

elif model_option == "VAR":
    tab1, tab2 = st.tabs(["🔮 Prediction", "📘 Method Explanation"])
    with tab1:
        df_forecast = run_var_model(data)
        st.line_chart(df_forecast.tail(100))
    with tab2:
        components.html(load_html("explanations_html/var.html"), height=600, scrolling=True)


elif model_option == "XGBoost":
    tab1, tab2, tab3 = st.tabs(["🔮 Prediction", "📊 Model Performance", "📘 Method Explanation"])

    with tab1:
        model, df_forecast, metrics = run_xgboost_model(data)
        st.line_chart(df_forecast[["Close", "XGB_Forecast"]])

    with tab2:
        st.metric("RMSE", f"{metrics['RMSE']:.4f}")
        st.metric("MAE", f"{metrics['MAE']:.4f}")
        st.metric("MAPE (%)", f"{metrics['MAPE']:.2f}%")

    with tab3:
        components.html(load_html("explanations_html/xgboost.html"), height=600, scrolling=True)


elif model_option == "Random Forest":
    tab1, tab2, tab3 = st.tabs(["🔮 Prediction", "📊 Model Performance", "📘 Method Explanation"])
    with tab1:
        model, df_forecast, metrics = run_random_forest_model(data)
        st.line_chart(df_forecast[["Close", "RF_Forecast"]])

    with tab2:
        st.metric("RMSE", f"{metrics['RMSE']:.4f}")
        st.metric("MAE", f"{metrics['MAE']:.4f}")
        st.metric("MAPE (%)", f"{metrics['MAPE']:.2f}%")
        
    with tab3:
        components.html(load_html("explanations_html/random_forest.html"), height=600, scrolling=True)

elif model_option == "LSTM":
    tab1, tab2, tab3 = st.tabs(["🔮 Prediction", "📊 Model Performance", "📘 Method Explanation"])
    with tab1:
        model, df_forecast, metrics = run_LSTM_model(data)
        st.line_chart(df_forecast[["Close", "RF_Forecast"]])

    with tab2:
        st.metric("RMSE", f"{metrics['RMSE']:.4f}")
        st.metric("MAE", f"{metrics['MAE']:.4f}")
        st.metric("MAPE (%)", f"{metrics['MAPE']:.2f}%")
        
    with tab3:
        components.html(load_html("explanations_html/lstm.html"), height=600, scrolling=True)




# elif model_option == "XGBoost":
#     tab1, tab2, tab3 = st.tabs(["🔮 Prediction", "📊 Model Performance", "📘 Method Explanation"])

#     with tab1:
#         with st.spinner("Training / predicting XGBoost..."):
#             df_forecast, metrics, model = run_xgboost_model(data, n_lags=8)
#         # show last 100 points: actual Close and predicted price (predictions will appear only for test tail)
#         plot_df = df_forecast[['Close', 'XGB_Pred_Price']].tail(200).reset_index(drop=True)
#         st.line_chart(plot_df)

#     with tab2:
#         st.write("Model performance on test set")
#         st.metric("RMSE", f"{metrics['RMSE']:.4f}")
#         st.metric("MAE", f"{metrics['MAE']:.4f}")
#         st.metric("MAPE (%)", f"{metrics['MAPE']:.2f}%")
#         st.write(f"Train / Val / Test sizes: {metrics['n_train']} / {metrics['n_val']} / {metrics['n_test']}")

#     with tab3:
#         components.html(load_html("explanations_html/xgboost.html"), height=600, scrolling=True)

