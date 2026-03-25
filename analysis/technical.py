"""
Technical indicator computation — pure pandas, no external TA libraries.
All indicators are appended as new columns to a copy of the input DataFrame.
"""

import pandas as pd
import numpy as np


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all technical indicators on an OHLCV DataFrame.

    Input columns required: Open, High, Low, Close, Volume
    Returns a new DataFrame with all indicator columns added.
    """
    df = df.copy()
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    # ------------------------------------------------------------------
    # Simple Moving Averages
    # ------------------------------------------------------------------
    df["SMA_20"] = close.rolling(window=20).mean()
    df["SMA_50"] = close.rolling(window=50).mean()
    df["SMA_200"] = close.rolling(window=200).mean()

    # ------------------------------------------------------------------
    # Exponential Moving Averages
    # ------------------------------------------------------------------
    df["EMA_12"] = close.ewm(span=12, adjust=False).mean()
    df["EMA_26"] = close.ewm(span=26, adjust=False).mean()

    # ------------------------------------------------------------------
    # Bollinger Bands (20-period, 2 std dev)
    # ------------------------------------------------------------------
    bb_middle = close.rolling(window=20).mean()
    bb_std = close.rolling(window=20).std()
    df["BB_middle"] = bb_middle
    df["BB_upper"] = bb_middle + 2 * bb_std
    df["BB_lower"] = bb_middle - 2 * bb_std

    # ------------------------------------------------------------------
    # MACD (12/26/9)
    # ------------------------------------------------------------------
    df["MACD_line"] = df["EMA_12"] - df["EMA_26"]
    df["MACD_signal"] = df["MACD_line"].ewm(span=9, adjust=False).mean()
    df["MACD_histogram"] = df["MACD_line"] - df["MACD_signal"]

    # ------------------------------------------------------------------
    # RSI (14-period)
    # ------------------------------------------------------------------
    df["RSI"] = _compute_rsi(close, period=14)

    # ------------------------------------------------------------------
    # Average True Range (14-period)
    # ------------------------------------------------------------------
    df["ATR"] = _compute_atr(high, low, close, period=14)

    # ------------------------------------------------------------------
    # Volume Moving Average
    # ------------------------------------------------------------------
    df["VOL_SMA_20"] = volume.rolling(window=20).mean()

    # ------------------------------------------------------------------
    # On-Balance Volume
    # ------------------------------------------------------------------
    df["OBV"] = _compute_obv(close, volume)

    # ------------------------------------------------------------------
    # Stochastic Oscillator (%K and %D, 14-period)
    # ------------------------------------------------------------------
    df["Stoch_K"], df["Stoch_D"] = _compute_stochastic(high, low, close, period=14, smooth_k=3, smooth_d=3)

    return df


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Use Wilder's smoothing (EWM with alpha=1/period)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, float("nan"))
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    return atr


def _compute_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = close.diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    obv = (direction * volume).cumsum()
    return obv


def _compute_stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
    smooth_k: int = 3,
    smooth_d: int = 3,
) -> tuple[pd.Series, pd.Series]:
    lowest_low = low.rolling(window=period).min()
    highest_high = high.rolling(window=period).max()
    denom = (highest_high - lowest_low).replace(0, float("nan"))
    raw_k = 100 * (close - lowest_low) / denom
    stoch_k = raw_k.rolling(window=smooth_k).mean()
    stoch_d = stoch_k.rolling(window=smooth_d).mean()
    return stoch_k, stoch_d


# ---------------------------------------------------------------------------
# Signal interpretation
# ---------------------------------------------------------------------------

def interpret_signals(df: pd.DataFrame) -> list[dict]:
    """
    Return a list of signal dicts for the Signal Summary table.
    Each dict: {indicator, value, signal, implication, color}
    """
    last = df.iloc[-1]
    signals = []

    # RSI
    rsi_val = last.get("RSI")
    if pd.notna(rsi_val):
        if rsi_val < 30:
            sig, impl, color = "Oversold", "Potential reversal upward", "green"
        elif rsi_val > 70:
            sig, impl, color = "Overbought", "Potential reversal downward", "red"
        else:
            sig, impl, color = "Neutral", "No extreme reading", "gray"
        signals.append({"Indicator": "RSI (14)", "Value": f"{rsi_val:.1f}", "Signal": sig, "Implication": impl, "color": color})

    # MACD
    macd_line = last.get("MACD_line")
    macd_signal = last.get("MACD_signal")
    macd_hist = last.get("MACD_histogram")
    if pd.notna(macd_hist):
        prev_hist = df["MACD_histogram"].iloc[-2] if len(df) > 1 else None
        if macd_line > macd_signal and (prev_hist is not None and prev_hist < 0 and macd_hist > 0):
            sig, impl, color = "Bullish crossover", "MACD crossed above signal", "green"
        elif macd_line < macd_signal and (prev_hist is not None and prev_hist > 0 and macd_hist < 0):
            sig, impl, color = "Bearish crossover", "MACD crossed below signal", "red"
        elif macd_line > macd_signal:
            sig, impl, color = "Bullish", "MACD above signal line", "green"
        else:
            sig, impl, color = "Bearish", "MACD below signal line", "red"
        signals.append({"Indicator": "MACD (12/26/9)", "Value": f"{macd_hist:.3f}", "Signal": sig, "Implication": impl, "color": color})

    # Bollinger Bands
    close_val = last.get("Close")
    bb_upper = last.get("BB_upper")
    bb_lower = last.get("BB_lower")
    bb_mid = last.get("BB_middle")
    if pd.notna(bb_upper) and pd.notna(bb_lower):
        bb_range = bb_upper - bb_lower
        if bb_range > 0:
            bb_pos = (close_val - bb_lower) / bb_range
        else:
            bb_pos = 0.5
        if bb_pos > 0.9:
            sig, impl, color = "Near upper band", "Potentially overbought", "red"
        elif bb_pos < 0.1:
            sig, impl, color = "Near lower band", "Potentially oversold", "green"
        else:
            sig, impl, color = "Within bands", "Price within normal range", "gray"
        signals.append({"Indicator": "Bollinger Bands", "Value": f"Pos: {bb_pos:.2f}", "Signal": sig, "Implication": impl, "color": color})

    # SMA 20
    sma_20 = last.get("SMA_20")
    if pd.notna(sma_20) and pd.notna(close_val):
        if close_val > sma_20:
            sig, impl, color = "Above SMA20", "Short-term bullish trend", "green"
        else:
            sig, impl, color = "Below SMA20", "Short-term bearish trend", "red"
        signals.append({"Indicator": "SMA 20", "Value": f"{sma_20:.2f}", "Signal": sig, "Implication": impl, "color": color})

    # SMA 50
    sma_50 = last.get("SMA_50")
    if pd.notna(sma_50) and pd.notna(close_val):
        if close_val > sma_50:
            sig, impl, color = "Above SMA50", "Medium-term bullish trend", "green"
        else:
            sig, impl, color = "Below SMA50", "Medium-term bearish trend", "red"
        signals.append({"Indicator": "SMA 50", "Value": f"{sma_50:.2f}", "Signal": sig, "Implication": impl, "color": color})

    # SMA 200
    sma_200 = last.get("SMA_200")
    if pd.notna(sma_200) and pd.notna(close_val):
        if close_val > sma_200:
            sig, impl, color = "Above SMA200", "Long-term bullish trend", "green"
        else:
            sig, impl, color = "Below SMA200", "Long-term bearish trend", "red"
        signals.append({"Indicator": "SMA 200", "Value": f"{sma_200:.2f}", "Signal": sig, "Implication": impl, "color": color})

    # Stochastic
    stoch_k = last.get("Stoch_K")
    stoch_d = last.get("Stoch_D")
    if pd.notna(stoch_k):
        if stoch_k < 20:
            sig, impl, color = "Oversold", "Stochastic in oversold zone", "green"
        elif stoch_k > 80:
            sig, impl, color = "Overbought", "Stochastic in overbought zone", "red"
        else:
            sig, impl, color = "Neutral", "Stochastic in neutral zone", "gray"
        signals.append({"Indicator": "Stochastic %K/%D", "Value": f"K:{stoch_k:.1f} D:{stoch_d:.1f}" if pd.notna(stoch_d) else f"K:{stoch_k:.1f}", "Signal": sig, "Implication": impl, "color": color})

    # ATR
    atr_val = last.get("ATR")
    if pd.notna(atr_val) and pd.notna(close_val) and close_val > 0:
        atr_pct = (atr_val / close_val) * 100
        if atr_pct > 3:
            sig, impl, color = "High Volatility", "Large price swings expected", "red"
        elif atr_pct < 1:
            sig, impl, color = "Low Volatility", "Price moving in tight range", "gray"
        else:
            sig, impl, color = "Normal Volatility", "Typical price movement", "gray"
        signals.append({"Indicator": "ATR (14)", "Value": f"{atr_val:.2f} ({atr_pct:.1f}%)", "Signal": sig, "Implication": impl, "color": color})

    return signals
