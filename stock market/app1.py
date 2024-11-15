import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

# Load the pre-trained LSTM model
model = load_model(r"C:\Users\behra\Downloads\stock market\Stock Predictions Model.keras")

# Streamlit interface
st.header('Stock Market Predictor')

# User input for stock symbol
stock = st.text_input('Enter Stock Symbol', 'GOOG')

# Set the dynamic date range for the last 20 years
end = datetime.now()
start = datetime(end.year - 20, end.month, end.day)

# Fetch stock data from Yahoo Finance
data = yf.download(stock, start, end)

# Display the stock data
st.subheader('Stock Data')
st.write(data)

# Prepare training and testing datasets
data_train = pd.DataFrame(data['Close'][0:int(len(data) * 0.80)])
data_test = pd.DataFrame(data['Close'][int(len(data) * 0.80):])

# Scaling the data for LSTM
scaler = MinMaxScaler(feature_range=(0, 1))
past_100_days = data_train.tail(100)
data_test_combined = pd.concat([past_100_days, data_test], ignore_index=True)
data_test_scaled = scaler.fit_transform(data_test_combined)

# Visualization: Price vs Moving Averages
st.subheader('Price vs Moving Averages')

# Moving averages calculation
ma_50 = data['Close'].rolling(window=50).mean()
ma_100 = data['Close'].rolling(window=100).mean()
ma_200 = data['Close'].rolling(window=200).mean()

# Plotting the Price and Moving Averages
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(data['Close'], label='Price', color='green')
ax.plot(ma_50, label='50-day MA', color='red')
ax.plot(ma_100, label='100-day MA', color='blue')
ax.plot(ma_200, label='200-day MA', color='orange')
plt.title('Stock Price vs Moving Averages')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig)

# Prepare input data for prediction
X_test, y_test = [], []
for i in range(100, data_test_scaled.shape[0]):
    X_test.append(data_test_scaled[i-100:i])
    y_test.append(data_test_scaled[i, 0])

X_test, y_test = np.array(X_test), np.array(y_test)

# Make predictions using the loaded model
predictions = model.predict(X_test)

# Rescaling predictions back to original prices
scale_factor = 1 / scaler.scale_[0]
predictions_rescaled = predictions * scale_factor
y_test_rescaled = y_test * scale_factor

# Plot Original Price vs Predicted Price
st.subheader('Original Price vs Predicted Price')
fig2, ax2 = plt.subplots(figsize=(10, 6))
ax2.plot(y_test_rescaled, label='Original Price', color='green')
ax2.plot(predictions_rescaled, label='Predicted Price', color='red')
plt.title('Original Price vs Predicted Price')
plt.xlabel('Time Steps')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)

# Future Prediction for the Next 504 Days
st.subheader('Future Price Prediction (Next 2 Years)')

# Use the last 100 days from the scaled test set as input for prediction
last_sequence = data_test_scaled[-100:]
last_sequence = np.reshape(last_sequence, (1, last_sequence.shape[0], 1))

# Predict future stock prices for 504 days
future_predictions = []
for _ in range(504):
    next_pred = model.predict(last_sequence)[0][0]
    future_predictions.append(next_pred)
    last_sequence = np.append(last_sequence[:, 1:, :], [[next_pred]], axis=1)

# Rescale the future predictions back to original prices
future_predictions_rescaled = np.array(future_predictions) * scale_factor

# Generate future dates for plotting, starting from the next day
last_date = data.index[-1]
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=504)

# Plot the future predictions
fig3, ax3 = plt.subplots(figsize=(10, 6))
ax3.plot(future_dates, future_predictions_rescaled, label='Future Price (Next 2 Years)', color='blue')
plt.title('Future Price Predictions')
plt.xlabel('Date')
plt.ylabel('Predicted Price')
plt.legend()
st.pyplot(fig3)
