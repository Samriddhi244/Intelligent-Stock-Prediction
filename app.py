
from flask import Flask, render_template, request, jsonify
from predict import predict_stock
import yfinance as yf
from datetime import datetime

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    ticker = request.form['ticker'].upper()
    print(f"Predicting for ticker: {ticker}")
    
    try:
        # Get current stock info for additional context
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Get prediction
        prediction_value = predict_stock(ticker)
        
        # Ensure prediction is a proper float value
        if prediction_value is None:
            raise ValueError("Prediction returned None value")
        
        # Convert to standard Python float to avoid numpy type issues
        prediction = float(prediction_value)
        
        # Get current price for comparison
        current_data = stock.history(period="1d")
        current_price = None
        if not current_data.empty:
            current_price = round(float(current_data['Close'].iloc[-1]), 2)
        
        # Calculate prediction vs current price
        price_change = None
        price_change_percent = None
        if current_price:
            price_change = round(prediction - current_price, 2)
            price_change_percent = round((price_change / current_price) * 100, 2)
        
        # Calculate dynamic confidence metrics
        # Base confidence on data quality and volatility
        historical_data = stock.history(period="60d")
        confidence = 85  # Base confidence
        error_margin = 3.5  # Base error margin
        
        if not historical_data.empty:
            # Calculate volatility-based adjustments
            volatility = historical_data['Close'].pct_change().std() * 100
            if volatility < 2:
                confidence += 5  # Lower volatility = higher confidence
                error_margin -= 0.5
            elif volatility > 5:
                confidence -= 10  # Higher volatility = lower confidence
                error_margin += 1.5
            
            # Adjust based on data completeness
            data_completeness = len(historical_data) / 60
            confidence = int(confidence * data_completeness)
            error_margin = round(error_margin / data_completeness, 1)
            
            # Ensure reasonable bounds
            confidence = max(60, min(95, confidence))
            error_margin = max(1.0, min(8.0, error_margin))
        
        return render_template('index.html', 
                             prediction=round(prediction, 2), 
                             ticker=ticker,
                             current_price=current_price,
                             price_change=price_change,
                             price_change_percent=price_change_percent,
                             company_name=info.get('longName', ticker),
                             sector=info.get('sector', 'N/A'),
                             confidence=confidence,
                             error_margin=error_margin,
                             success=True,
                             error=False)
                             
    except ValueError as ve:
        print(f"ValueError: {ve}")
        # More specific error messages for common issues
        error_msg = str(ve)
        if "No data found for ticker" in error_msg:
            error_msg = f"Unable to find stock data for ticker '{ticker}'. Please check if the ticker symbol is correct."
        elif "No valid data found after filtering" in error_msg:
            error_msg = f"Data for '{ticker}' exists but contains no valid trading information."
        elif "Insufficient data" in error_msg:
            error_msg = f"Not enough historical data for '{ticker}' to make a prediction (need at least 60 trading days)."
        return render_template('index.html', prediction=f"Data Error: {error_msg}", ticker=ticker, success=False, error=True)
        
    except Exception as e:
        print(f"General Error: {e}")
        return render_template('index.html', prediction=f"Error: {str(e)}", ticker=ticker, success=False, error=True)

@app.route('/api/stock-info/<ticker>')
def get_stock_info(ticker):
    """API endpoint to get basic stock information"""
    try:
        stock = yf.Ticker(ticker.upper())
        info = stock.info
        current_data = stock.history(period="5d")
        
        if current_data.empty:
            return jsonify({"error": "No data found"}), 404
            
        return jsonify({
            "symbol": ticker.upper(),
            "name": info.get('longName', ticker),
            "sector": info.get('sector', 'N/A'),
            "current_price": round(current_data['Close'].iloc[-1], 2),
            "volume": int(current_data['Volume'].iloc[-1]),
            "market_cap": info.get('marketCap', 'N/A')
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
