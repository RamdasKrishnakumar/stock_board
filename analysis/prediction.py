"""
ML-based next-day directional prediction using RandomForestClassifier.
Features are engineered from the indicator-enriched DataFrame produced by
analysis/technical.py.
"""

import hashlib
import warnings

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add ML feature columns to *df* and return a copy."""
    df = df.copy()
    close = df["Close"]
    volume = df["Volume"]

    # Price-based
    df["daily_return"] = close.pct_change()
    df["log_return"] = np.log(close / close.shift(1))
    df["price_vs_SMA20"] = (close - df["SMA_20"]) / df["SMA_20"]
    df["price_vs_SMA50"] = (close - df["SMA_50"]) / df["SMA_50"]
    bb_range = (df["BB_upper"] - df["BB_lower"]).replace(0, np.nan)
    df["BB_position"] = (close - df["BB_lower"]) / bb_range
    df["ATR_pct"] = df["ATR"] / close

    # Momentum
    # RSI, MACD_histogram, Stoch_K, Stoch_D already present

    # Volume
    df["volume_change_pct"] = volume.pct_change()
    df["OBV_change"] = df["OBV"].diff()
    df["volume_vs_avg"] = volume / df["VOL_SMA_20"].replace(0, np.nan)

    # Lag features
    for lag in [1, 2, 3, 5]:
        df[f"return_lag_{lag}"] = df["daily_return"].shift(lag)

    # Rolling volatility
    df["rolling_volatility_5d"] = df["daily_return"].rolling(5).std()
    df["rolling_volatility_20d"] = df["daily_return"].rolling(20).std()

    # Target: 1 if tomorrow's close > today's close
    df["target"] = (close.shift(-1) > close).astype(int)

    return df


_FEATURE_COLS = [
    "daily_return", "log_return", "price_vs_SMA20", "price_vs_SMA50",
    "BB_position", "ATR_pct",
    "RSI", "MACD_histogram", "Stoch_K", "Stoch_D",
    "volume_change_pct", "OBV_change", "volume_vs_avg",
    "return_lag_1", "return_lag_2", "return_lag_3", "return_lag_5",
    "rolling_volatility_5d", "rolling_volatility_20d",
]


def _dataframe_hash(df: pd.DataFrame) -> str:
    """Produce a short hash of the DataFrame to use as a cache key."""
    return hashlib.md5(
        pd.util.hash_pandas_object(df, index=True).values.tobytes()
    ).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Model training + inference (cached by ticker + data hash)
# ---------------------------------------------------------------------------

@st.cache_resource
def _train_model(ticker: str, data_hash: str, feature_df_json: str):
    """
    Train a RandomForestClassifier and return (model, scaler, test_accuracy,
    feature_importances).  Arguments are hashable so Streamlit can cache by value.
    """
    df = pd.read_json(feature_df_json)

    # Ensure we have enough data
    if len(df) < 80:
        return None, None, None, None

    feature_df = df[_FEATURE_COLS + ["target"]].dropna()

    if len(feature_df) < 80:
        return None, None, None, None

    # Train/test split: last 60 rows = test set
    TEST_SIZE = 60
    train = feature_df.iloc[:-TEST_SIZE]
    test = feature_df.iloc[-TEST_SIZE:]

    X_train, y_train = train[_FEATURE_COLS], train["target"]
    X_test, y_test = test[_FEATURE_COLS], test["target"]

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=10,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train_sc, y_train)

    y_pred = model.predict(X_test_sc)
    accuracy = float(accuracy_score(y_test, y_pred))

    importances = dict(
        sorted(
            zip(_FEATURE_COLS, model.feature_importances_),
            key=lambda x: x[1],
            reverse=True,
        )[:10]
    )

    return model, scaler, accuracy, importances


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def predict(indicator_df: pd.DataFrame, ticker: str = "UNKNOWN") -> dict | None:
    """
    Run the full ML pipeline on *indicator_df* (already enriched by
    analysis/technical.compute_indicators).

    Returns
    -------
    dict with keys:
        prediction  : "UP" | "DOWN"
        confidence  : float (0–1)
        accuracy    : float (0–1) test-set accuracy
        feature_importance : dict name → importance (top 10)
        signals     : dict of current indicator signal labels
    or None if not enough data.
    """
    if len(indicator_df) < 80:
        return None

    # Engineer features
    feat_df = _engineer_features(indicator_df)

    # Drop the last row from training data (target is NaN because we shifted by -1)
    train_data = feat_df.iloc[:-1]
    data_hash = _dataframe_hash(train_data)

    with st.spinner("Training prediction model…"):
        model, scaler, accuracy, importances = _train_model(
            ticker, data_hash, train_data.to_json()
        )

    if model is None:
        return None

    # Predict on the most recent complete row
    latest_features = feat_df[_FEATURE_COLS].dropna()
    if latest_features.empty:
        return None

    latest_row = latest_features.iloc[[-1]]
    latest_scaled = scaler.transform(latest_row)
    proba = model.predict_proba(latest_scaled)[0]  # [P(DOWN), P(UP)]

    predicted_class = int(model.predict(latest_scaled)[0])
    prediction = "UP" if predicted_class == 1 else "DOWN"
    confidence = float(proba[predicted_class])

    # Current indicator signals
    signals = _build_signals(indicator_df)

    return {
        "prediction": prediction,
        "confidence": confidence,
        "accuracy": accuracy,
        "feature_importance": importances,
        "signals": signals,
    }


# ---------------------------------------------------------------------------
# Signal helpers
# ---------------------------------------------------------------------------

def _build_signals(df: pd.DataFrame) -> dict:
    """Build a plain-English signal dict from the latest row of indicators."""
    last = df.iloc[-1]
    signals = {}

    rsi = last.get("RSI")
    if pd.notna(rsi):
        if rsi < 30:
            signals["RSI"] = "Oversold"
        elif rsi > 70:
            signals["RSI"] = "Overbought"
        else:
            signals["RSI"] = "Neutral"

    macd_line = last.get("MACD_line")
    macd_sig = last.get("MACD_signal")
    if pd.notna(macd_line) and pd.notna(macd_sig):
        if len(df) > 1:
            prev_macd = df["MACD_line"].iloc[-2]
            prev_sig = df["MACD_signal"].iloc[-2]
            if prev_macd < prev_sig and macd_line > macd_sig:
                signals["MACD"] = "Bullish crossover"
            elif prev_macd > prev_sig and macd_line < macd_sig:
                signals["MACD"] = "Bearish crossover"
            elif macd_line > macd_sig:
                signals["MACD"] = "Bullish"
            else:
                signals["MACD"] = "Bearish"

    bb_upper = last.get("BB_upper")
    bb_lower = last.get("BB_lower")
    close = last.get("Close")
    if pd.notna(bb_upper) and pd.notna(bb_lower) and pd.notna(close):
        bb_range = bb_upper - bb_lower
        if bb_range > 0:
            pos = (close - bb_lower) / bb_range
            if pos > 0.9:
                signals["BB"] = "Near upper band"
            elif pos < 0.1:
                signals["BB"] = "Near lower band"
            else:
                signals["BB"] = "Within bands"

    stoch_k = last.get("Stoch_K")
    if pd.notna(stoch_k):
        if stoch_k < 20:
            signals["Stochastic"] = "Oversold"
        elif stoch_k > 80:
            signals["Stochastic"] = "Overbought"
        else:
            signals["Stochastic"] = "Neutral"

    sma_20 = last.get("SMA_20")
    if pd.notna(sma_20) and pd.notna(close):
        signals["SMA 20"] = "Price above SMA20" if close > sma_20 else "Price below SMA20"

    sma_50 = last.get("SMA_50")
    if pd.notna(sma_50) and pd.notna(close):
        signals["SMA 50"] = "Price above SMA50" if close > sma_50 else "Price below SMA50"

    atr = last.get("ATR")
    if pd.notna(atr) and pd.notna(close) and close > 0:
        atr_pct = (atr / close) * 100
        if atr_pct > 3:
            signals["Volatility (ATR)"] = "High volatility"
        elif atr_pct < 1:
            signals["Volatility (ATR)"] = "Low volatility"
        else:
            signals["Volatility (ATR)"] = "Normal volatility"

    return signals
