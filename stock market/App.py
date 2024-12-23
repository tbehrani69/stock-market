{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "f761aa28-64dc-40d8-a007-7a7e134ac411",
   "metadata": {},
   "outputs": [],
   "source": [
    "import streamlit as st\n",
    "import plotly.graph_objects as go\n",
    "from datetime import datetime, timedelta\n",
    "import yfinance as yf\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "from sklearn.preprocessing import MinMaxScaler\n",
    "from tensorflow.keras.models import Sequential\n",
    "from tensorflow.keras.layers import LSTM, Dense"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "e87a03c1-553e-4095-a8dd-13a4a691c889",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Set dark theme\n",
    "st.set_page_config(\n",
    "    page_title=\"Stock Price Prediction\",\n",
    "    layout=\"wide\",\n",
    "    initial_sidebar_state=\"collapsed\"\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "271901f0-05ca-408d-86d7-a2130d020409",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2024-11-16 04:35:04.060 \n",
      "  \u001b[33m\u001b[1mWarning:\u001b[0m to view this Streamlit app on a browser, run it with the following\n",
      "  command:\n",
      "\n",
      "    streamlit run D:\\anaconda\\Lib\\site-packages\\ipykernel_launcher.py [ARGUMENTS]\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "DeltaGenerator()"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Custom CSS for dark theme\n",
    "st.markdown(\"\"\"\n",
    "<style>\n",
    ".stApp {\n",
    "    background-color: #202124;\n",
    "    color: #ffffff;\n",
    "}\n",
    ".stock-header {\n",
    "    font-size: 2.5em;\n",
    "    font-weight: bold;\n",
    "    margin-bottom: 0.5em;\n",
    "}\n",
    ".stock-price {\n",
    "    font-size: 2em;\n",
    "    font-weight: bold;\n",
    "}\n",
    ".price-change {\n",
    "    color: #ff4444;\n",
    "    font-size: 1.2em;\n",
    "}\n",
    ".metric-card {\n",
    "    background-color: #303134;\n",
    "    padding: 1em;\n",
    "    border-radius: 10px;\n",
    "    margin: 0.5em;\n",
    "}\n",
    ".tab-container {\n",
    "    display: flex;\n",
    "    gap: 1em;\n",
    "    margin-bottom: 1em;\n",
    "}\n",
    ".tab {\n",
    "    background-color: #303134;\n",
    "    padding: 0.5em 2em;\n",
    "    border-radius: 20px;\n",
    "    cursor: pointer;\n",
    "}\n",
    ".tab.active {\n",
    "    background-color: #8ab4f8;\n",
    "    color: #202124;\n",
    "}\n",
    "</style>\n",
    "\"\"\", unsafe_allow_html=True)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "5e5482bf-324a-4584-9c4b-cd757a7437c2",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Function to fetch stock data\n",
    "def fetch_stock_data(symbol, start_date, end_date):\n",
    "    stock = yf.Ticker(symbol)\n",
    "    df = stock.history(start=start_date, end=end_date)\n",
    "    return df[['Open', 'High', 'Low', 'Close', 'Volume']]\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "2ffa2621-6779-431c-917f-397d9cc3994b",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Function to prepare stock data for LSTM model\n",
    "def prepare_stock_data(symbol):\n",
    "    # Fetch data for the past 5 years\n",
    "    end_date = datetime.today().date()\n",
    "    start_date = end_date - timedelta(days=5*365)\n",
    "    df = fetch_stock_data(symbol, start_date, end_date)\n",
    "    \n",
    "    # Normalize the data\n",
    "    scaler = MinMaxScaler()\n",
    "    scaled_data = scaler.fit_transform(df[['Open', 'High', 'Low', 'Close', 'Volume']])\n",
    "    \n",
    "    return df, scaled_data, scaler"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "128a2fca-f992-420b-86dc-6e2ba9e9fbfb",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Function to create LSTM model\n",
    "def create_lstm_model(data, time_steps, future_days):\n",
    "    # Normalize the data to a range between 0 and 1\n",
    "    scaler = MinMaxScaler()\n",
    "    scaled_data = scaler.fit_transform(data)\n",
    "    \n",
    "    # Prepare the data for LSTM with the given time steps\n",
    "    X, y = [], []\n",
    "    for i in range(time_steps, len(scaled_data) - future_days + 1):\n",
    "        X.append(scaled_data[i - time_steps:i])\n",
    "        y.append(scaled_data[i:i + future_days, 3])  # '3' is for 'Close' column index\n",
    "    \n",
    "    # Convert to numpy arrays for model training\n",
    "    X, y = np.array(X), np.array(y)\n",
    "    n_features = X.shape[2]\n",
    "    \n",
    "    # Define the LSTM model structure\n",
    "    model = Sequential([\n",
    "        LSTM(50, return_sequences=True, input_shape=(time_steps, n_features)),\n",
    "        LSTM(50, return_sequences=False),\n",
    "        Dense(future_days)\n",
    "    ])\n",
    "    model.compile(optimizer='adam', loss='mse')\n",
    "    \n",
    "    # Train the model\n",
    "    model.fit(X, y, epochs=50, batch_size=32, validation_split=0.1, verbose=1)\n",
    "    \n",
    "    return model, scaler"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "96318f46-944d-453d-b209-d81946957bba",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Function to predict future prices\n",
    "def predict_future_prices(model, scaler, data, time_steps, future_days):\n",
    "    scaled_data = scaler.transform(data[['Open', 'High', 'Low', 'Close', 'Volume']])\n",
    "    X_pred = []\n",
    "    for i in range(len(scaled_data) - time_steps, len(scaled_data)):\n",
    "        X_pred.append(scaled_data[i - time_steps:i])\n",
    "    \n",
    "    X_pred = np.array(X_pred)\n",
    "    predictions = model.predict(X_pred)\n",
    "    \n",
    "    # Inverse transform the predictions\n",
    "    predicted_scaled = scaler.inverse_transform(np.column_stack((predictions, np.zeros((future_days, 4)))))\n",
    "    predicted_prices = predicted_scaled[:, 3]\n",
    "    \n",
    "    return predicted_prices"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "251bd0fa-ab39-4bc7-a888-72a7f0fe30fb",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "DeltaGenerator()"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Streamlit app\n",
    "st.title('Stock Market Prediction App')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "ffaf15e1-e45b-4a45-977b-1ce3a4d8260d",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2024-11-16 04:36:17.729 Session state does not function when running a script without `streamlit run`\n"
     ]
    }
   ],
   "source": [
    "# Input for stock symbol\n",
    "symbol = st.text_input(\"Enter Stock Symbol (e.g., AAPL, MSFT, AMZN):\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "da7180ab-bade-4217-bf4a-888b32f609e2",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Common stock symbols\n",
    "common_symbols = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'FB', 'TSLA', 'NVDA', 'JPM', 'JNJ', 'V']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "260bd15e-94a3-40fd-a637-00fae90c73cc",
   "metadata": {},
   "outputs": [],
   "source": [
    "if symbol:\n",
    "    try:\n",
    "        # Check if the symbol needs a suffix\n",
    "        if '.' not in symbol and symbol.isalpha():\n",
    "            symbol = f\"{symbol}\"  # Add .NS suffix for Indian stocks\n",
    "\n",
    "        # Fetch current stock data\n",
    "        stock = yf.Ticker(symbol)\n",
    "        current_data = stock.history(period='1d')\n",
    "\n",
    "        if current_data.empty:\n",
    "            st.error(f\"No data found for symbol {symbol}. Please check the symbol and try again.\")\n",
    "            st.info(f\"Here are some common stock symbols you can try: {', '.join(common_symbols)}\")\n",
    "        else:\n",
    "            current_price = current_data['Close'].iloc[-1]\n",
    "            price_change = current_data['Close'].iloc[-1] - current_data['Open'].iloc[0]\n",
    "            price_change_percent = (price_change / current_data['Open'].iloc[0]) * 100\n",
    "\n",
    "            # Display stock header and current price\n",
    "            st.markdown(f\"<div class='stock-header'>{stock.info.get('longName', symbol)}</div>\", unsafe_allow_html=True)\n",
    "            st.markdown(f\"<div class='stock-price'>{current_price:.2f} {stock.info.get('currency', 'USD')}</div>\", unsafe_allow_html=True)\n",
    "            st.markdown(f\"<div class='price-change'>{price_change:.2f} ({price_change_percent:.2f}%) today</div>\", unsafe_allow_html=True)\n",
    "\n",
    "            # Time period selector\n",
    "            periods = ['1D', '5D', '1M', '6M', 'YTD', 'More']\n",
    "            selected_period = st.select_slider('Select Time Period', options=periods, value='1D')\n",
    "\n",
    "            # Fetch and display stock data based on selected period\n",
    "            if selected_period == '1D':\n",
    "                historical_data = stock.history(interval='5m', period='1d')\n",
    "            elif selected_period == '5D':\n",
    "                historical_data = stock.history(interval='15m', period='5d')\n",
    "            elif selected_period == '1M':\n",
    "                historical_data = stock.history(interval='1d', period='1mo')\n",
    "            elif selected_period == '6M':\n",
    "                historical_data = stock.history(interval='1d', period='6mo')\n",
    "            else:\n",
    "                historical_data = stock.history(interval='1d', period='ytd')\n",
    "\n",
    "            # Create price chart\n",
    "            fig = go.Figure()\n",
    "            fig.add_trace(go.Scatter(\n",
    "                x=historical_data.index,\n",
    "                y=historical_data['Close'],\n",
    "                mode='lines',\n",
    "                name='Price',\n",
    "                line=dict(color='blue')\n",
    "            ))\n",
    "            st.plotly_chart(fig, use_container_width=True)\n",
    "\n",
    "            # Prediction button\n",
    "            if st.button('Show Price Prediction'):\n",
    "                st.write(\"Making Predictions...\")\n",
    "                df, scaled_data, scaler = prepare_stock_data(symbol)\n",
    "                model, scaler = create_lstm_model(df, time_steps=60, future_days=7)\n",
    "                predictions = predict_future_prices(model, scaler, df, time_steps=60, future_days=7)\n",
    "                \n",
    "                st.write(\"Predicted Prices for the Next 7 Days:\")\n",
    "                st.write(predictions)\n",
    "\n",
    "    except Exception as e:\n",
    "        st.error(f\"Error fetching data for {symbol}: {str(e)}\")\n",
    "else:\n",
    "    st.info(\"Please enter a stock symbol to start.\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "946140e7-2681-4257-b4b3-057e74a8a858",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
