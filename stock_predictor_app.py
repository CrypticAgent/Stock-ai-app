
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.title("🔮 AI Stock Predictor")
ticker = st.sidebar.text_input("Stock ticker (e.g., TSLA)", "TSLA").upper()
period = st.sidebar.number_input("Years of historical data",  start=1, min_value=1, max_value=5)

if st.sidebar.button("Predict"):
    data = yf.download(ticker, period=f"{period}y")
    st.write(f"Showing last {len(data)} trading days for {ticker}")
    st.line_chart(data["Close"])

    df = data.copy()
    df['MA10'] = df['Close'].rolling(10).mean()
    df['MA50'] = df['Close'].rolling(50).mean()
    df['VolumeChange'] = df['Volume'].pct_change()
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df.dropna(inplace=True)

    features = ['MA10', 'MA50', 'VolumeChange']
    X, y = df[features], df['Target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    st.write(f"Test Accuracy: **{acc*100:.2f}%**")

    latest = X.iloc[-1].values.reshape(1, -1)
    pred = model.predict(latest)[0]
    prob = model.predict_proba(latest)[0]
    signal = "BUY 📈" if pred == 1 else "SELL / HOLD 📉"
    conf = f"{prob[pred]*100:.1f}% confidence"

    st.subheader(f"➡️ Model Suggests: **{signal}**")
    st.write(conf)
