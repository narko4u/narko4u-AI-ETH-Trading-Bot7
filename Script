import os
import time
import numpy as np
import pandas as pd
import ta
import ccxt
from sklearn.neighbors import KNeighborsClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# 🔐 Load Binance API keys from Railway environment variables
binance = ccxt.binance({
    'apiKey': os.getenv('BINANCE_API_KEY'),
    'secret': os.getenv('BINANCE_SECRET_KEY'),
    'options': {'defaultType': 'future'}  # Enable Futures Trading
})

# 📌 Trading Configuration
symbol = 'ETH/USDT'   # ✅ Trading Ethereum
timeframe = '15m'      # ✅ 15-minute timeframe
leverage = 5           # ✅ Set leverage (adjustable)
trade_size = 0.05      # ✅ Position size (ETH)
risk_factor = 2        # ✅ ATR multiplier for stop-loss/take-profit

# 🔥 Fetch Historical Data
def fetch_data(symbol, timeframe='15m', limit=500):
    ohlcv = binance.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

# 📊 Add Technical Indicators
def add_indicators(df):
    df['EMA_50'] = ta.trend.ema_indicator(df['close'], window=50)
    df['EMA_200'] = ta.trend.ema_indicator(df['close'], window=200)
    df['MACD'] = ta.trend.macd(df['close'])
    df['RSI'] = ta.momentum.rsi(df['close'], window=14)
    df['ATR'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
    df.dropna(inplace=True)
    return df

# 🤖 kNN Pattern Recognition
def train_knn(df):
    df['Target'] = np.where(df['close'].shift(-1) > df['close'], 1, 0)
    features = ['close', 'EMA_50', 'EMA_200', 'MACD', 'RSI', 'ATR']
    X = df[features].values
    y = df['Target'].values
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X[:-1], y[:-1])  # Train on all but the last row
    return knn

# 🔮 LSTM Model for Price Prediction
def create_lstm_model():
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(10, 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# 📈 Train LSTM Model
def train_lstm(df):
    X, y = [], []
    data = df['close'].values.reshape(-1, 1)
    for i in range(len(data) - 10):
        X.append(data[i:i+10])
        y.append(data[i+10])
    X, y = np.array(X), np.array(y)
    lstm_model = create_lstm_model()
    lstm_model.fit(X, y, epochs=10, batch_size=16, verbose=0)
    return lstm_model

# 🔍 Predict Next Move
def predict_next_move(df, knn, lstm_model):
    last_features = df[['close', 'EMA_50', 'EMA_200', 'MACD', 'RSI', 'ATR']].values[-1].reshape(1, -1)
    knn_prediction = knn.predict(last_features)[0]
    last_sequence = df['close'].values[-10:].reshape(1, 10, 1)
    lstm_prediction = lstm_model.predict(last_sequence)[0][0]

    return knn_prediction, lstm_prediction

# 🛑 Calculate Dynamic Stop-Loss & Take-Profit
def calculate_risk_levels(df, risk_factor=2):
    atr = df['ATR'].iloc[-1]
    stop_loss = df['close'].iloc[-1] - (risk_factor * atr)
    take_profit = df['close'].iloc[-1] + (risk_factor * atr)
    return stop_loss, take_profit

# 💰 Execute Trades
def execute_trade(symbol, size, side):
    order = binance.create_market_order(symbol, side, size, {'type': 'future'})
    return order

# 📜 Trading Bot Execution
df = fetch_data(symbol)
df = add_indicators(df)
knn = train_knn(df)
lstm_model = train_lstm(df)

print(f"✅ Trading Bot Started for {symbol}")

while True:
    try:
        df = fetch_data(symbol)
        df = add_indicators(df)
        
        knn_pred, lstm_pred = predict_next_move(df, knn, lstm_model)
        stop_loss, take_profit = calculate_risk_levels(df, risk_factor)

        current_price = df['close'].iloc[-1]
        position_size = trade_size * leverage  # Dynamic leverage applied
        
        if knn_pred == 1 and lstm_pred > current_price:
            print(f"🔵 BUY Signal for {symbol} at {current_price}")
            execute_trade(symbol, position_size, 'buy')

        elif knn_pred == 0 and lstm_pred < current_price:
            print(f"🔴 SELL Signal for {symbol} at {current_price}")
            execute_trade(symbol, position_size, 'sell')

        print(f"🎯 Stop-Loss: {stop_loss}, Take-Profit: {take_profit}")
        time.sleep(60)  # Wait before the next trade cycle

    except Exception as e:
        print(f"❌ Error: {e}")
        time.sleep(30)
