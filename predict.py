
import yfinance as yf
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

# Load model and scaler
model = load_model('model/model.h5')
with open('model/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

def predict_stock(ticker):
    try:
        # Download data with better error handling
        df = yf.download(ticker, start='2012-01-01', end=datetime.now(), progress=False)
        
        # Check if data is available
        if df.empty:
            raise ValueError(f"No data found for ticker {ticker}")
        
        print(f"Downloaded data for {ticker}. Shape: {df.shape}")
        print(f"Available columns: {df.columns.tolist()}")
        
        # Remove any multi-level column indexing that yfinance sometimes creates
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        # Ensure we have the basic required columns
        required_columns = ['Close']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Use the same features that were likely used during training
        # Most models use OHLCV data or just Close prices
        # Let's try with multiple features first, then fall back to Close only
        
        try:
            # Try with OHLCV features (common in stock prediction models)
            available_ohlcv = ['Open', 'High', 'Low', 'Close', 'Volume']
            ohlcv_cols = [col for col in available_ohlcv if col in df.columns]
            
            if len(ohlcv_cols) >= 5:  # All OHLCV columns available
                data = df[ohlcv_cols].dropna()
                dataset = data.values
                print(f"Using OHLCV features: {ohlcv_cols}. Data shape: {dataset.shape}")
                
                if dataset.shape[0] == 0:
                    raise ValueError("No valid OHLCV data after removing NaN values")
                
                scaled_data = scaler.transform(dataset)
                print("Scaled data shape:", scaled_data.shape)
            else:
                raise Exception("Not all OHLCV columns available")
            
        except Exception as e1:
            print(f"OHLCV approach failed: {e1}")
            try:
                # Try with just OHLC features
                available_ohlc = ['Open', 'High', 'Low', 'Close']
                ohlc_cols = [col for col in available_ohlc if col in df.columns]
                
                if len(ohlc_cols) >= 4:  # All OHLC columns available
                    data = df[ohlc_cols].dropna()
                    dataset = data.values
                    print(f"Using OHLC features: {ohlc_cols}. Data shape: {dataset.shape}")
                    
                    if dataset.shape[0] == 0:
                        raise ValueError("No valid OHLC data after removing NaN values")
                    
                    scaled_data = scaler.transform(dataset)
                    print("Scaled data shape:", scaled_data.shape)
                else:
                    raise Exception("Not all OHLC columns available")
                
            except Exception as e2:
                print(f"OHLC approach failed: {e2}")
                # Fall back to Close only but ensure proper shape
                if 'Close' in df.columns:
                    data = df[['Close']].dropna()
                    dataset = data.values
                    print(f"Using Close only. Data shape: {dataset.shape}")
                    
                    # Check if we have valid data
                    if dataset.shape[0] == 0:
                        raise ValueError("No valid Close price data after removing NaN values")
                    if dataset.shape[1] == 0:
                        raise ValueError("Close column is empty")
                    
                    # Handle scaler feature dimension mismatch
                    try:
                        scaled_data = scaler.transform(dataset)
                        print("Scaled data shape:", scaled_data.shape)
                    except ValueError as scaler_error:
                        if "feature" in str(scaler_error).lower():
                            print(f"Scaler dimension mismatch: {scaler_error}")
                            # Create dummy features to match scaler expectations
                            scaler_features = scaler.n_features_in_
                            print(f"Scaler expects {scaler_features} features, we have {dataset.shape[1]}")
                            
                            if scaler_features > 1:
                                # Pad with zeros or duplicate the Close price
                                padded_dataset = np.zeros((dataset.shape[0], scaler_features))
                                padded_dataset[:, 0] = dataset[:, 0]  # Close price in first column
                                # Fill other columns with Close price (simple approach)
                                for i in range(1, scaler_features):
                                    padded_dataset[:, i] = dataset[:, 0]
                                dataset = padded_dataset
                                print(f"Padded dataset shape: {dataset.shape}")
                            
                            scaled_data = scaler.transform(dataset)
                            print("Scaled data shape after padding:", scaled_data.shape)
                        else:
                            raise scaler_error
                else:
                    raise ValueError("Close column not found in the data")
    
    except Exception as download_error:
        raise ValueError(f"Failed to download or process data for {ticker}: {str(download_error)}")

    # Ensure we have enough data for the prediction window
    if len(scaled_data) < 60:
        raise ValueError(f"Insufficient data: only {len(scaled_data)} days available, need at least 60")

    last_60_days = scaled_data[-60:]
    X_test = np.array([last_60_days])
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    pred_price = model.predict(X_test)
    
    # For inverse transform, we need to handle the feature dimension properly
    if pred_price.shape[1] != scaled_data.shape[1]:
        # If prediction is single value but scaler expects multiple features
        # Create a dummy array with the same number of features
        dummy_pred = np.zeros((pred_price.shape[0], scaled_data.shape[1]))
        dummy_pred[:, 0] = pred_price[:, 0]  # Assume first feature is the target
        pred_price = scaler.inverse_transform(dummy_pred)
        return pred_price[0][0]
    else:
        pred_price = scaler.inverse_transform(pred_price)
        return pred_price[0][0]
