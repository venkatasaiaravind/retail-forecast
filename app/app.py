import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from prophet import Prophet
from scipy.optimize import minimize_scalar
from sklearn.metrics import mean_squared_error
import plotly.express as px

# -------------------------------
#  Streamlit UI Setup
# -------------------------------
st.set_page_config(page_title="Retail Forecast + Price Optimization", layout="centered")
st.title("Retail Demand Forecast & Price Optimization")

st.markdown("Forecast future sales using Prophet and find the **optimal price** to maximize revenue.")

# -------------------------------
#  Load Dataset
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv('../data/train.csv', low_memory=False, dtype={'StateHoliday': str})
    df['Date'] = pd.to_datetime(df['Date'])
    return df

df = load_data()

# Sidebar - Store selector
store_id = st.sidebar.selectbox("Select Store ID", sorted(df['Store'].unique()))

# Filter for selected store
store_df = df[df['Store'] == store_id]

# Prepare data for Prophet
daily_sales = store_df.groupby('Date')['Sales'].sum().reset_index()
daily_sales.columns = ['ds', 'y']

# -------------------------------
#  Prophet Forecasting
# -------------------------------
model = Prophet()
model.fit(daily_sales)

# Make future predictions
future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)

# Evaluate RMSE
actual = daily_sales['y'][-30:].values
predicted = forecast['yhat'][-30:].values[:30]
rmse = np.sqrt(mean_squared_error(actual, predicted))
st.metric(label="Forecast RMSE", value=f"{rmse:.2f}")

# -------------------------------
#  Simplified Forecast Plot (150 Days)
# -------------------------------
st.subheader("Simplified Forecast: Last 120 Days + Next 30 Days")

last_actual_date = daily_sales['ds'].max()
forecast_trimmed = forecast[forecast['ds'] > (last_actual_date - pd.Timedelta(days=120))]
actual_trimmed = daily_sales[daily_sales['ds'] > (last_actual_date - pd.Timedelta(days=120))]
actual_trimmed['Type'] = 'Actual'
forecast_only = forecast_trimmed[forecast_trimmed['ds'] > last_actual_date]
forecast_only = forecast_only[['ds', 'yhat']].rename(columns={'yhat': 'y'})
forecast_only['Type'] = 'Forecast'

combined = pd.concat([actual_trimmed[['ds', 'y', 'Type']], forecast_only])

fig_simple = px.line(
    combined, x='ds', y='y', color='Type',
    title=f"Store {store_id} – Last 120 Days + Forecast (Next 30 Days)",
    labels={'ds': 'Date', 'y': 'Sales'},
    template="plotly_white"
)
fig_simple.update_traces(line=dict(width=2))
st.plotly_chart(fig_simple, use_container_width=True)

# -------------------------------
#  Price Optimization Section
# -------------------------------
st.subheader("Price Optimization")

st.markdown("Estimate the best product price to **maximize revenue** using a simulated demand curve.")

# Price sliders
min_price = st.slider("Minimum Price (Rupees)", 50, 200, 100)
max_price = st.slider("Maximum Price (Rupees)", min_price + 10, 1000, 500)

# Simulate demand
prices = np.arange(min_price, max_price, 10)
demand = 2000 - 3 * prices + np.random.normal(0, 50, len(prices))
revenue = prices * demand

# Optimize revenue
def revenue_function(p):
    d = 2000 - 3 * p
    return -(p * d)

res = minimize_scalar(revenue_function, bounds=(min_price, max_price), method='bounded')
optimal_price = res.x
max_revenue = -res.fun

# Show optimized values
st.success(f"Optimal Price: {optimal_price:.2f}(Rupees)")
st.success(f"Expected Max Revenue: {max_revenue:.2f}(Rupees)")

# Revenue Curve using Plotly
import plotly.graph_objects as go

st.subheader("Revenue vs Price (Interactive Plot)")

fig_revenue = go.Figure()

# Line plot for revenue
fig_revenue.add_trace(go.Scatter(
    x=prices,
    y=revenue,
    mode='lines+markers',
    name='Revenue Curve',
    line=dict(color='blue', width=3)
))

# Vertical line for optimal price
fig_revenue.add_vline(
    x=optimal_price,
    line=dict(color='red', width=2, dash='dash'),
    annotation_text=f"Optimal Rupees: {optimal_price:.2f}",
    annotation_position="top right"
)

fig_revenue.update_layout(
    title="Revenue vs Price Optimization",
    xaxis_title="Price (Rupees)",
    yaxis_title="Revenue (Rupees)",
    template="plotly_white",
    showlegend=True
)

st.plotly_chart(fig_revenue, use_container_width=True)

st.markdown("---")
st.header("📘 Project Summary & Component Explanation")

st.markdown("""
### 🧠 1. Sales Forecasting
We used **Facebook Prophet**, a time series forecasting model, to predict daily store sales for the next 30 days. Prophet considers:
- Daily, weekly, yearly trends
- Holidays and seasonality
- Historical sales patterns

The graph above shows **actual sales** (last 60 days) and **predicted sales** (next 30 days), clearly separated for easy comparison.

---

### 💸 2. Price Optimization
Using a **simulated price-demand curve**, we calculate:
- Demand drops as price increases
- Revenue = Price × Demand
- Using optimization (`scipy.optimize`), we find the **price point that maximizes revenue**

This is helpful for **retail decision-makers** to decide **how much to charge** for maximum profit.

---

### 📊 3. Graphs
- The **forecast graph** is built with Plotly to be interactive, zoomable, and readable. It compares historical data with model predictions.
- The **revenue curve** shows how changing the product price affects revenue and highlights the **optimal price point** visually.

---

### 📝 Notes
- This is a prototype. In a real-world setting, price-demand would come from real transaction data or A/B testing.
- Prophet can be further improved with holiday data and external regressors like promotions, competitors, etc.

""")
