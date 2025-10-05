import pandas as pd
import numpy as np
import streamlit as st
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_components_plotly
from scipy.optimize import minimize_scalar
import plotly.graph_objects as go
from pathlib import Path
import itertools

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(page_title="Automated Retail Forecasting", layout="wide")
st.title("Automated Retail Forecast & Optimization Suite 📊")
st.markdown("Select a store to automatically receive the most accurate, fine-tuned forecast and price optimization.")

# -------------------------------
# Data Loading & Preparation (Cached)
# -------------------------------
@st.cache_data
def load_data(train_path: Path, test_path: Path) -> (pd.DataFrame, pd.DataFrame):
    """Loads, preprocesses, and handles data types."""
    try:
        train_df = pd.read_csv(train_path, low_memory=False, parse_dates=['Date'])
        test_df = pd.read_csv(test_path, low_memory=False, parse_dates=['Date'])
        test_df['Open'] = test_df['Open'].fillna(1).astype(int)
        return train_df, test_df
    except FileNotFoundError as e:
        st.error(f"Error: `{e.filename}` not found. Please ensure it's in the same directory as the app.")
        st.stop()

# -------------------------------
# Holiday Modeling (Cached)
# -------------------------------
@st.cache_data
def create_holidays_df(df: pd.DataFrame) -> pd.DataFrame:
    """Creates a DataFrame of holidays for Prophet."""
    state_holidays = df[df['StateHoliday'] != '0'][['Date', 'StateHoliday']]
    state_holidays = state_holidays.rename(columns={'Date': 'ds', 'StateHoliday': 'holiday'}).drop_duplicates()
    return state_holidays

# -------------------------------
# Automated Hyperparameter Tuning (Cached)
# -------------------------------
@st.cache_data
def tune_hyperparameters(store_data: pd.DataFrame, holidays_df: pd.DataFrame) -> dict:
    """
    Automatically finds the best Prophet parameters for a given store's data.
    """
    param_grid = {  
        'changepoint_prior_scale': [0.01, 0.1, 0.5],
        'seasonality_prior_scale': [1.0, 10.0, 20.0],
    }
    all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
    
    rmses = []

    prophet_df = store_data[['Date', 'Sales', 'Promo', 'SchoolHoliday']].rename(columns={'Date': 'ds', 'Sales': 'y'})

    for params in all_params:
        m = Prophet(holidays=holidays_df, **params)
        m.add_regressor('Promo')
        m.add_regressor('SchoolHoliday')
        m.fit(prophet_df)
        df_cv = cross_validation(m, initial='180 days', period='90 days', horizon = '30 days', parallel=None)
        df_p = performance_metrics(df_cv, rolling_window=1)
        rmses.append(df_p['rmse'].values[0])

    best_params = all_params[np.argmin(rmses)]
    return best_params

# -------------------------------
# Final Forecasting Function
# -------------------------------
@st.cache_data
def run_final_forecast(store_data: pd.DataFrame, holidays_df: pd.DataFrame, best_params: dict) -> (Prophet, pd.DataFrame, pd.DataFrame):
    """
    Fits the final model using the best parameters and runs cross-validation for metrics.
    """
    prophet_df = store_data[['Date', 'Sales', 'Promo', 'SchoolHoliday']].rename(columns={'Date': 'ds', 'Sales': 'y'})
    model = Prophet(holidays=holidays_df, **best_params)
    model.add_regressor('Promo')
    model.add_regressor('SchoolHoliday')
    model.fit(prophet_df)

    future = model.make_future_dataframe(periods=30)
    future['Promo'] = prophet_df['Promo'].iloc[-1]
    future['SchoolHoliday'] = prophet_df['SchoolHoliday'].iloc[-1]
    forecast = model.predict(future)

    df_cv = cross_validation(model, initial='180 days', period='90 days', horizon='30 days', parallel=None)
    metrics_df = performance_metrics(df_cv, rolling_window=1)
    
    return model, forecast, metrics_df

# --- Main App Execution ---
train_path = Path("train.csv")
test_path = Path("test.csv")
train_df, test_df = load_data(train_path, test_path)

st.sidebar.header("Store Selection")
store_id = st.sidebar.selectbox("Select Store ID", sorted(train_df['Store'].unique()))

store_data = train_df[(train_df['Store'] == store_id) & (train_df['Sales'] > 0)].copy()
holidays_df = create_holidays_df(train_df)

if not store_data.empty and len(store_data) > 210:
    with st.spinner(f"Auto-tuning model for Store {store_id}... This may take a minute."):
        best_params = tune_hyperparameters(store_data, holidays_df)
    
    model, forecast, metrics_df = run_final_forecast(store_data, holidays_df, best_params)

    st.sidebar.header("Optimal Model Settings")
    st.sidebar.metric("Best Trend Flexibility", f"{best_params['changepoint_prior_scale']:.2f}")
    st.sidebar.metric("Best Seasonality Strength", f"{best_params['seasonality_prior_scale']:.1f}")

    tab1, tab2, tab3, tab4 = st.tabs(["📊 Sales Forecast", "📈 Forecast Components", "💡 Promotion Impact", "⚙️ Generate Predictions"])

    with tab1:
        st.header(f"Sales Forecast & Price Optimization for Store {store_id}")
        
        with st.expander("View Final Model Performance"):
            mape_display = f"{metrics_df['mape'].values[0]:.2%}" if 'mape' in metrics_df else "N/A"
            mae, rmse = metrics_df['mae'].values[0], metrics_df['rmse'].values[0]
            c1, c2, c3 = st.columns(3)
            c1.metric("Forecast Error (MAPE)", mape_display)
            c2.metric("Forecast Error (MAE)", f"{mae:,.0f}")
            c3.metric("Forecast Error (RMSE)", f"{rmse:,.0f}")
        
        fig_forecast = go.Figure()
        fig_forecast.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Predicted Sales', line=dict(color='orange')))
        fig_forecast.add_trace(go.Scatter(x=store_data['Date'], y=store_data['Sales'], name='Actual Sales', mode='lines', line=dict(color='blue'), opacity=0.6))
        fig_forecast.update_layout(title="Forecast vs. Actual Sales", xaxis_title="Date", yaxis_title="Sales")
        st.plotly_chart(fig_forecast, use_container_width=True)

        st.header("Dynamic Price Optimization")
        col1, col2 = st.columns([1, 2])
        with col1:
            avg_forecasted_demand = forecast['yhat'][-30:].mean()
            st.metric("Avg. Forecasted Daily Demand", f"{avg_forecasted_demand:.0f} units")
            baseline_price = st.slider("Current Price (P₀)", 50, 1000, 350, 10)
            elasticity = st.slider("Price Elasticity", -3.0, -0.5, -1.5, 0.1)

            lower_bound, upper_bound = baseline_price * 0.5, baseline_price * 2.0

            def calculate_revenue(price, p0, q0, ped):
                if p0 == 0: return 0
                new_demand = q0 * (1 + ped * ((price - p0) / p0))
                return -(price * max(0, new_demand))

            res = minimize_scalar(calculate_revenue, bounds=(lower_bound, upper_bound), args=(baseline_price, avg_forecasted_demand, elasticity), method='bounded')
            st.success(f"**Optimal Price: ₹{res.x:.2f}**")
            st.info(f"**Expected Max Revenue: ₹{-res.fun:,.2f}**")
        
        with col2:
            prices = np.linspace(lower_bound, upper_bound, 200)
            revenues = [-calculate_revenue(p, baseline_price, avg_forecasted_demand, elasticity) for p in prices]
            fig_revenue = go.Figure()
            fig_revenue.add_trace(go.Scatter(x=prices, y=revenues, mode='lines', name='Revenue'))
            fig_revenue.add_vline(x=res.x, line_dash="dash", line_color="red", annotation_text="Optimal")
            fig_revenue.update_layout(title="Revenue vs. Price Curve", xaxis_title="Price (₹)", yaxis_title="Revenue (₹)")
            st.plotly_chart(fig_revenue, use_container_width=True)

    with tab2:
        st.header("Forecast Components")
        fig_components = plot_components_plotly(model, forecast)
        st.plotly_chart(fig_components, use_container_width=True)

    with tab3:
        st.header("Promotion Impact Analysis")
        promo_impact = forecast['Promo'][forecast['Promo'] != 0].mean()
        
        # FIX: Handle cases where promo impact is not a number
        if pd.isna(promo_impact):
            promo_impact = 0.0

        st.metric("Estimated Sales Boost from a Promotion", f"{promo_impact:.2f} units")
        st.markdown("""
        This value represents the model's estimation of the average increase in daily sales when a promotion is active.
        A value of 0 indicates that promotions were not found to be a significant driver of sales for this store.
        """)

        promo_sales = store_data[store_data['Promo'] == 1]['Sales'].mean()
        non_promo_sales = store_data[store_data['Promo'] == 0]['Sales'].mean()
        fig_promo = go.Figure(data=[go.Bar(name='Non-Promo Days', x=['Average Sales'], y=[non_promo_sales]), go.Bar(name='Promo Days', x=['Average Sales'], y=[promo_sales])])
        fig_promo.update_layout(barmode='group', title='Average Daily Sales: Promo vs. Non-Promo Days')
        st.plotly_chart(fig_promo, use_container_width=True)

    with tab4:
        st.header("Generate Submission File for Test Data")
        if st.button("Generate Prediction File"):
            with st.spinner("Forecasting for all stores... This may take several minutes."):
                open_test_df = test_df[test_df['Open'] == 1]
                all_predictions = []
                for s_id in open_test_df['Store'].unique():
                    store_train_data = train_df[(train_df['Store'] == s_id) & (train_df['Sales'] > 0)]
                    if not store_train_data.empty:
                        best_s_params = tune_hyperparameters(store_train_data, holidays_df)
                        m = Prophet(holidays=holidays_df, **best_s_params)
                        m.add_regressor('Promo')
                        m.add_regressor('SchoolHoliday')
                        m.fit(store_train_data.rename(columns={'Date': 'ds', 'Sales': 'y'}))
                        future_df = open_test_df[open_test_df['Store'] == s_id][['Date', 'Promo', 'SchoolHoliday']].rename(columns={'Date': 'ds'})
                        if not future_df.empty:
                            predictions = m.predict(future_df)
                            predictions['Id'] = open_test_df[open_test_df['Store'] == s_id]['Id']
                            all_predictions.append(predictions[['Id', 'yhat']])
                if all_predictions:
                    final_preds = pd.concat(all_predictions).rename(columns={'yhat': 'Sales'})
                    submission_df = test_df[['Id']].merge(final_preds, on='Id', how='left').fillna(0)
                    submission_df['Sales'] = submission_df['Sales'].astype(int)
                    submission_df.sort_values('Id', inplace=True)
                    csv = submission_df.to_csv(index=False).encode('utf-8')
                    st.success("Prediction file generated!")
                    st.download_button("Download submission.csv", csv, 'submission.csv', 'text/csv')
else:
    st.warning(f"Store ID {store_id} has insufficient data for analysis.")