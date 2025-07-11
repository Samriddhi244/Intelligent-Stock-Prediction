
# 📈 Intelligent Stock Price Prediction Using LSTM

This project predicts the future closing price of a stock using historical market data and an LSTM (Long Short-Term Memory) neural network. The trained model is deployed using Flask and allows users to input any stock ticker to receive the predicted next-day closing price.

---

## 📂 Project Structure

```
stock_prediction_app/
│
├── app.py                  # Flask backend application
├── predict.py              # Contains the prediction logic
├── requirements.txt        # Required dependencies
│
├── model/
│   ├── model.h5            # Trained LSTM model
│   └── scaler.pkl          # Scaler used for data normalization
│
├── templates/
│   └── index.html          # Frontend UI for stock prediction
```

---

## 🔧 How It Works

1. **User inputs stock ticker** on the web interface.
2. Flask backend fetches **latest historical stock data** using `yfinance`.
3. The **MinMaxScaler** is applied (same one used during training).
4. Data is reshaped and passed to the **trained LSTM model**.
5. Model returns a **predicted stock closing price** for the next day.
6. The result is displayed back on the web UI.

---

## 📊 Machine Learning Model

- **Model Type:** LSTM (Long Short-Term Memory)
- **Framework:** TensorFlow / Keras
- **Training Data:** Historical daily closing prices from Yahoo Finance
- **Preprocessing:**
  - Normalization using `MinMaxScaler`
  - Windowing data: 60 previous days → 1 predicted day
- **Loss Function:** Mean Squared Error (MSE)
- **RMSE Achieved:** ~6.97 (on AAPL stock data)

---

## 🧠 Technologies Used

| Library/Module         | Purpose |
|------------------------|---------|
| `pandas`               | Data manipulation and analysis |
| `numpy`                | Numerical operations |
| `matplotlib`, `seaborn`| Data visualization (EDA phase) |
| `yfinance`             | Fetching stock data from Yahoo Finance |
| `sklearn`              | Scaling data (`MinMaxScaler`) |
| `keras` / `tensorflow` | LSTM model creation and training |
| `pickle`               | Saving/loading the scaler object |
| `flask`                | Web framework for building the app |
| `datetime`             | Handling time ranges for stock data |

---

## 🚀 Getting Started

### ✅ Prerequisites

Install dependencies using:

```bash
pip install -r requirements.txt
```

### ▶️ Run the App

```bash
python app.py
```

Open your browser and go to: [http://localhost:5000](http://localhost:5000)

---

## 📈 Example Usage

Input: `AAPL`  
Output: `Predicted Price: $203.45`

---

## 🌟 Features

- Real-time prediction for any stock supported by Yahoo Finance
- LSTM-based forecasting model
- Extendable to support multiple-step predictions or multi-feature input

---

## 🔄 Future Enhancements

- Predict next 7-day prices
- Add historical vs. predicted price charts
- Include additional features (e.g., volume, open price)
- Deploy to cloud platforms like Render, Railway, or Vercel

---

## ❓ FAQs

**Q: Can I use this for stocks like TSLA or NIFTY50?**  
Yes, just input a valid Yahoo Finance ticker like `TSLA`, `^NSEI`, or `INFY.NS`.

**Q: How do I re-train with more data?**  
Change the `start` parameter in `yfinance.download()` and re-train the model.

---

## 👨‍💻 Author

**Samriddhi Bhalekar**  
*Computer Science (AI & ML)*

---

## 📄 License

This project is open-source and free to use for educational purposes.
