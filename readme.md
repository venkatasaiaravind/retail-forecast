# 📈 Retail Demand Forecasting & Price Optimization

This Streamlit web application uses Facebook Prophet to forecast store sales for the next 30 days and simulates an optimal pricing strategy to maximize revenue.

---

## 🚀 Features

- **Store-wise Forecasting** using Prophet
- **Interactive Sales Forecast Visualization** (Actual vs. Predicted)
- **Revenue vs. Price Optimization Curve**
- **Optimal Price Recommendation** using `scipy.optimize`
- **Kaggle-style submission CSV** generation (optional extension)

---

## 🛠️ Tech Stack

- Python
- Streamlit
- Prophet (Facebook)
- Pandas, NumPy, Plotly, Matplotlib
- Scikit-learn, SciPy

---

## 📂 Project Structure

```
smartretail_project/
├── app/
│   └── app.py                # Main Streamlit app code
├── data/
│   ├── train.csv
├── requirements.txt         # Required Python packages
└── README.md                # You're here
```

---

## 📦 Installation & Running

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/retail-forecast.git
cd retail-forecast
```

### 2. Create Environment & Install Dependencies

```bash
conda create -n smartretail python=3.10
conda activate smartretail
pip install -r requirements.txt
```

### 3. Launch the App

```bash
streamlit run app/app.py
```

---

## 📊 Model Description

### 🔮 Prophet Forecasting

- Daily sales aggregated by date
- Model fits to historical data
- Forecasts next 30 days
- RMSE shown for last 30-day prediction vs. actuals

### 💸 Price Optimization

- Simulated demand: `D = 2000 - 3 * Price`
- Revenue: `Price * Demand`
- Uses `scipy.optimize.minimize_scalar()` to find the best price point

---

## 📷 Screenshots

| Forecast Graph | Revenue Optimization |
| -------------- | -------------------- |
|                |                      |

---

## 📌 Deployment

Deployed via [Streamlit Cloud](https://share.streamlit.io)

- Upload code to GitHub
- Go to Streamlit Cloud → New App
- Set `app/app.py` as entry point

---

## 🧠 Author

**Aravind**\
B.Tech CSE | Data Science Enthusiast\
Hyderabad Institute of Technology and Management

---

## 📄 License

This project is open-source and free to use under the MIT license.

