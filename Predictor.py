import sys
import warnings
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import norm, pearsonr

# Suppress warnings
if not sys.warnoptions:
    warnings.simplefilter('ignore')

# Set random seed for reproducibility
sns.set()
tf.random.set_seed(1)
np.random.seed(1)

# Advanced Evaluation Metrics
def advanced_forecast_evaluation(actual, predictions):
    """
    Comprehensive forecast evaluation with multiple metrics
    
    Parameters:
    - actual: True price values
    - predictions: Array of predicted price scenarios
    
    Returns:
    Dictionary of evaluation metrics
    """
    metrics = {}
    
    # Directional Accuracy
    actual_changes = np.diff(actual)
    predicted_changes = np.diff(predictions.mean(axis=0))
    direction_accuracy = np.mean(np.sign(actual_changes) == np.sign(predicted_changes)) * 100
    
    # Confidence Interval Coverage
    ci_lower = np.percentile(predictions, 2.5, axis=0)
    ci_upper = np.percentile(predictions, 97.5, axis=0)
    ci_coverage = np.mean((actual >= ci_lower) & (actual <= ci_upper)) * 100
    
    # Prediction Interval Width
    interval_width = np.mean(ci_upper - ci_lower)
    
    # Correlation Metrics
    prediction_mean = predictions.mean(axis=0)
    pearson_corr, _ = pearsonr(actual, prediction_mean)
    
    # Volatility Comparison
    actual_volatility = np.std(actual)
    predicted_volatility = np.mean(np.std(predictions, axis=1))
    
    metrics = {
        'trend_accuracy': direction_accuracy,
        'confidence_interval_coverage': ci_coverage,
        'prediction_interval_width': interval_width,
        'correlation': pearson_corr,
        'actual_volatility': actual_volatility,
        'predicted_volatility': predicted_volatility
    }
    
    return metrics

def get_stock_data(ticker, start_date, end_date):
    """
    Retrieve stock data with additional preprocessing
    
    Parameters:
    - ticker: Stock symbol
    - start_date: Data start date
    - end_date: Data end date
    
    Returns:
    Processed DataFrame with additional features
    """
    import yfinance as yf
    
    stock = yf.Ticker(ticker)
    data = stock.history(start=start_date, end=end_date)
    
    # Advanced feature engineering
    data['Daily_Return'] = data['Close'].pct_change()
    data['Log_Return'] = np.log(data['Close'] / data['Close'].shift(1))
    
    # Multiple moving averages
    for window in [5, 20, 50]:
        data[f'MA_{window}'] = data['Close'].rolling(window=window).mean()
    
    # Exponential moving averages
    data['EMA_20'] = data['Close'].ewm(span=20, adjust=False).mean()
    
    # Volatility measures
    data['Rolling_Volatility'] = data['Daily_Return'].rolling(window=20).std() * np.sqrt(252)
    
    return data.dropna()

def create_sequences(data, sequence_length, features):
    """
    Create input sequences for LSTM model
    
    Parameters:
    - data: Input DataFrame
    - sequence_length: Number of timesteps
    - features: List of feature columns
    
    Returns:
    X, y sequences for model training
    """
    X, y = [], []
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[features])
    
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i])
        y.append(scaled_data[i, features.index('Close')])
    
    return np.array(X), np.array(y), scaler

def build_lstm_model(input_shape, dropout_rate=0.3):
    """
    Construct LSTM model with advanced architecture
    
    Parameters:
    - input_shape: Shape of input sequences
    
    Returns:
    Compiled TensorFlow model
    """
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(128, return_sequences=True, input_shape=input_shape),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(dropout_rate),
        
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(dropout_rate),
        
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
                  loss='mean_squared_error')
    
    return model

def monte_carlo_simulation(initial_price, num_days, num_simulations, volatility, drift):
    """
    Advanced Monte Carlo price simulation
    
    Parameters:
    - initial_price: Starting price
    - num_days: Simulation duration
    - num_simulations: Number of scenarios
    - volatility: Price volatility
    - drift: Price trend
    
    Returns:
    Simulated price paths
    """
    dt = 1 / 252  # Trading days in a year
    
    # Geometric Brownian Motion with more sophisticated parameters
    returns = np.random.normal(
        loc=(drift - 0.5 * volatility**2) * dt, 
        scale=volatility * np.sqrt(dt),
        size=(num_simulations, num_days)
    )
    
    price_paths = initial_price * np.exp(np.cumsum(returns, axis=1))
    return price_paths

def main():
    # ------ CHANGE STOCK AND DATES HERE -------
    ticker = 'AAPL'
    start_date = '2015-01-01'
    end_date = '2023-01-01'
    sequence_length = 20
    num_simulations = 1000
    
    # Retrieve and preprocess data
    df = get_stock_data(ticker, start_date, end_date)
    
    # Select advanced features
    features = ['Close', 'Volume', 'Daily_Return', 'Log_Return', 
                'MA_5', 'MA_20', 'EMA_20', 'Rolling_Volatility']
    
    # Prepare sequences
    X, y, scaler = create_sequences(df, sequence_length, features)
    
    # Train-test split
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Model training
    model = build_lstm_model(X_train.shape[1:])
    
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=10, 
        restore_best_weights=True
    )
    
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=100,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=0
    )
    
    # Predictions and simulations
    predictions = model.predict(X_test).flatten()
    
    # Inverse transform predictions
    predictions = scaler.inverse_transform(
        np.column_stack([predictions] + [np.zeros(len(predictions)) for _ in range(len(features)-1)])
    )[:, 0]
    
    # True test values
    true_values = df['Close'].iloc[-len(X_test):]
    
    # Monte Carlo simulation
    volatility = np.std(np.diff(np.log(df['Close']))) * np.sqrt(252)
    drift = np.mean(np.diff(np.log(df['Close']))) * 252
    
    mc_simulations = monte_carlo_simulation(
        initial_price=predictions[0], 
        num_days=len(predictions),
        num_simulations=num_simulations,
        volatility=volatility,
        drift=drift
    )
    
    # Advanced evaluation
    evaluation_metrics = advanced_forecast_evaluation(true_values.values, mc_simulations)
    
    # Visualization
    plt.figure(figsize=(15, 10))
    
    # Original vs Predicted
    plt.subplot(2, 1, 1)
    plt.plot(true_values, label='Actual Price', color='black')
    plt.plot(true_values.index, predictions, label='Predicted Price', color='blue')
    plt.title(f'{ticker} Price Prediction')
    plt.legend()
    
    # Monte Carlo Simulations
    plt.subplot(2, 1, 2)
    for i in range(50):  # Plot subset of simulations
        plt.plot(mc_simulations[i], alpha=0.1, color='blue')
    plt.plot(true_values, label='Actual Price', color='black')
    plt.title('Monte Carlo Price Simulations')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Print evaluation metrics
    print("Forecast Evaluation Metrics:")
    for metric, value in evaluation_metrics.items():
        print(f"{metric}: {value}")

if __name__ == "__main__":
    main()
