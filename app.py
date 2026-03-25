"""
StockPulse — Production-ready stock tracking & prediction dashboard.

HOW TO RUN LOCALLY:
    pip install -r requirements.txt
    streamlit run app.py

HOW TO DEPLOY FREE ON STREAMLIT COMMUNITY CLOUD:
    1. Push this entire folder to a GitHub repository.
    2. Go to https://share.streamlit.io and sign in with GitHub.
    3. Click "New app", select your repo, branch, and set main file to app.py.
    4. Click "Deploy" — your app will be live in ~60 seconds.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data.fetcher import fetch_price_history, fetch_company_info, fetch_news
from analysis.technical import compute_indicators, interpret_signals
from analysis.prediction import predict
from analysis.sentiment import analyze_sentiment

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="StockPulse",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Shared chart style
# ---------------------------------------------------------------------------
TEAL = "#00d4aa"
RED = "#ff4b6e"
CHART_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#fafafa"),
    margin=dict(l=10, r=10, t=40, b=10),
    xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.07)"),
    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.07)"),
)

PERIOD_MAP = {"1M": "1mo", "3M": "3mo", "6M": "6mo", "1Y": "1y", "2Y": "2y", "5Y": "5y"}

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Metric card tweaks */
    [data-testid="metric-container"] {
        background: #1a1f2e;
        border: 1px solid #2a2f3e;
        border-radius: 10px;
        padding: 12px 16px;
    }
    /* Prediction badge */
    .pred-badge {
        display: inline-block;
        padding: 18px 48px;
        border-radius: 14px;
        font-size: 2.6rem;
        font-weight: 800;
        letter-spacing: 2px;
        text-align: center;
    }
    .pred-up   { background: rgba(0,212,170,0.18); color: #00d4aa; border: 2px solid #00d4aa; }
    .pred-down { background: rgba(255,75,110,0.18); color: #ff4b6e; border: 2px solid #ff4b6e; }
    /* Sentiment badge */
    .sent-badge {
        display: inline-block;
        padding: 10px 28px;
        border-radius: 10px;
        font-size: 1.4rem;
        font-weight: 700;
    }
    .sent-pos  { background: rgba(0,212,170,0.18); color: #00d4aa; border: 2px solid #00d4aa; }
    .sent-neg  { background: rgba(255,75,110,0.18); color: #ff4b6e; border: 2px solid #ff4b6e; }
    .sent-neu  { background: rgba(180,180,180,0.12); color: #aaaaaa; border: 2px solid #555; }
    /* News article row */
    .news-row {
        background: #1a1f2e;
        border-radius: 10px;
        padding: 12px 16px;
        margin-bottom: 10px;
        border-left: 4px solid #2a2f3e;
    }
    .news-pos { border-left-color: #00d4aa; }
    .news-neg { border-left-color: #ff4b6e; }
    .news-neu { border-left-color: #555; }
    /* Signal table pill */
    .pill-green { background:#00d4aa22; color:#00d4aa; padding:3px 10px; border-radius:20px; font-weight:600; }
    .pill-red   { background:#ff4b6e22; color:#ff4b6e; padding:3px 10px; border-radius:20px; font-weight:600; }
    .pill-gray  { background:#55555522; color:#aaa;    padding:3px 10px; border-radius:20px; font-weight:600; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("📈 StockPulse")
    st.markdown("---")

    ticker_input = st.text_input("Ticker Symbol", value="AAPL", max_chars=10).strip().upper()
    period_label = st.radio("Period", list(PERIOD_MAP.keys()), index=3, horizontal=True)
    period = PERIOD_MAP[period_label]

    analyze_btn = st.button("🔍 Analyze", use_container_width=True, type="primary")

    st.markdown("---")
    st.markdown("**Chart Overlays**")
    show_sma = st.checkbox("Show SMAs", value=True)
    show_bb = st.checkbox("Show Bollinger Bands", value=True)
    show_volume = st.checkbox("Show Volume", value=True)

    st.markdown("---")
    st.caption("Data: Yahoo Finance · Model: RandomForest\nSentiment: VADER")

# ---------------------------------------------------------------------------
# Session state: persist last-analyzed results
# ---------------------------------------------------------------------------
if "analyzed" not in st.session_state:
    st.session_state.analyzed = False
if "ticker" not in st.session_state:
    st.session_state.ticker = ""

if analyze_btn:
    st.session_state.ticker = ticker_input
    st.session_state.analyzed = True

# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------
if not st.session_state.analyzed:
    st.markdown(
        """
        <div style="text-align:center; margin-top: 80px;">
            <h1 style="font-size:3rem; color:#00d4aa;">📈 StockPulse</h1>
            <p style="font-size:1.2rem; color:#aaa; max-width:500px; margin:auto;">
                Enter a ticker symbol in the sidebar and click <strong>Analyze</strong>
                to see real-time price data, technical indicators, ML-based directional
                predictions, and news sentiment.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

ticker = st.session_state.ticker

# ---------------------------------------------------------------------------
# Fetch all data
# ---------------------------------------------------------------------------
with st.spinner(f"Fetching data for {ticker}…"):
    raw_df = fetch_price_history(ticker, period=period)

if raw_df is None:
    st.stop()

with st.spinner("Computing technical indicators…"):
    df = compute_indicators(raw_df)

with st.spinner("Fetching company info…"):
    info = fetch_company_info(ticker)

with st.spinner("Fetching news…"):
    news = fetch_news(ticker, limit=20)

# ---------------------------------------------------------------------------
# Header — company name + metric row
# ---------------------------------------------------------------------------
company_name = info["name"] if info else ticker
st.markdown(f"## {company_name} &nbsp;<span style='color:#888;font-size:1rem'>({ticker})</span>", unsafe_allow_html=True)
if info:
    st.caption(f"{info.get('sector','N/A')} · {info.get('industry','N/A')}")

if info:
    cur = info.get("current_price")
    prev = info.get("previous_close")
    chg_pct = info.get("day_change_pct")
    chg_abs = (cur - prev) if (cur and prev) else None

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        st.metric("Current Price", f"${cur:.2f}" if cur else "N/A",
                  delta=f"{chg_abs:+.2f} ({chg_pct:+.2f}%)" if (chg_abs and chg_pct) else None)
    with col2:
        st.metric("Market Cap", info.get("market_cap", "N/A"))
    with col3:
        pe = info.get("pe_ratio")
        st.metric("P/E Ratio", f"{pe:.1f}" if pe else "N/A")
    with col4:
        st.metric("52W High", f"${info['week_52_high']:.2f}" if info.get("week_52_high") else "N/A")
    with col5:
        st.metric("52W Low", f"${info['week_52_low']:.2f}" if info.get("week_52_low") else "N/A")
    with col6:
        prev_close = info.get("previous_close")
        st.metric("Prev. Close", f"${prev_close:.2f}" if prev_close else "N/A")

st.markdown("---")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📉 Technical Analysis", "🤖 Prediction", "📰 Sentiment"])


# ============================================================
# TAB 1 — OVERVIEW
# ============================================================
with tab1:
    # Candlestick chart
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.75, 0.25],
    )

    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"],
        name="Price",
        increasing_line_color=TEAL,
        decreasing_line_color=RED,
    ), row=1, col=1)

    if show_sma:
        for col_name, color, width in [
            ("SMA_20", "#f7b731", 1.5),
            ("SMA_50", "#a29bfe", 1.5),
            ("SMA_200", "#fd9644", 1.5),
        ]:
            valid = df[col_name].dropna()
            if not valid.empty:
                fig.add_trace(go.Scatter(
                    x=df.index, y=df[col_name], name=col_name.replace("_", " "),
                    line=dict(color=color, width=width), opacity=0.85,
                ), row=1, col=1)

    if show_bb:
        valid_bb = df["BB_upper"].dropna()
        if not valid_bb.empty:
            fig.add_trace(go.Scatter(
                x=df.index, y=df["BB_upper"], name="BB Upper",
                line=dict(color="rgba(100,149,237,0.6)", width=1, dash="dot"),
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=df.index, y=df["BB_lower"], name="BB Lower",
                fill="tonexty", fillcolor="rgba(100,149,237,0.05)",
                line=dict(color="rgba(100,149,237,0.6)", width=1, dash="dot"),
            ), row=1, col=1)

    if show_volume:
        colors = [TEAL if c >= o else RED for c, o in zip(df["Close"], df["Open"])]
        fig.add_trace(go.Bar(
            x=df.index, y=df["Volume"], name="Volume",
            marker_color=colors, opacity=0.7,
        ), row=2, col=1)

    fig.update_layout(
        **CHART_LAYOUT,
        height=600,
        title=f"{ticker} Price History ({period_label})",
        showlegend=True,
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    if show_volume:
        fig.update_yaxes(title_text="Volume", row=2, col=1)

    st.plotly_chart(fig, use_container_width=True)

    # Company description
    if info and info.get("description"):
        with st.expander("About the Company"):
            st.write(info["description"])


# ============================================================
# TAB 2 — TECHNICAL ANALYSIS
# ============================================================
with tab2:
    # Full chart: candles + SMAs + BB + MACD + RSI + Stoch
    fig2 = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.5, 0.18, 0.16, 0.16],
        subplot_titles=("Price + Indicators", "MACD", "RSI (14)", "Stochastic %K/%D"),
    )

    # Candlestick
    fig2.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"], name="Price",
        increasing_line_color=TEAL, decreasing_line_color=RED,
    ), row=1, col=1)

    # All SMAs
    for col_name, color, lbl in [
        ("SMA_20", "#f7b731", "SMA 20"),
        ("SMA_50", "#a29bfe", "SMA 50"),
        ("SMA_200", "#fd9644", "SMA 200"),
        ("EMA_12", "#55efc4", "EMA 12"),
        ("EMA_26", "#00cec9", "EMA 26"),
    ]:
        if col_name in df.columns and df[col_name].notna().any():
            fig2.add_trace(go.Scatter(
                x=df.index, y=df[col_name], name=lbl,
                line=dict(color=color, width=1.2), opacity=0.8,
            ), row=1, col=1)

    # Bollinger Bands
    if df["BB_upper"].notna().any():
        fig2.add_trace(go.Scatter(
            x=df.index, y=df["BB_upper"], name="BB Upper",
            line=dict(color="rgba(100,149,237,0.5)", width=1, dash="dot"),
        ), row=1, col=1)
        fig2.add_trace(go.Scatter(
            x=df.index, y=df["BB_lower"], name="BB Lower",
            fill="tonexty", fillcolor="rgba(100,149,237,0.05)",
            line=dict(color="rgba(100,149,237,0.5)", width=1, dash="dot"),
        ), row=1, col=1)

    # MACD
    if df["MACD_line"].notna().any():
        fig2.add_trace(go.Scatter(
            x=df.index, y=df["MACD_line"], name="MACD",
            line=dict(color=TEAL, width=1.5),
        ), row=2, col=1)
        fig2.add_trace(go.Scatter(
            x=df.index, y=df["MACD_signal"], name="Signal",
            line=dict(color="#f7b731", width=1.5),
        ), row=2, col=1)
        hist_colors = [TEAL if v >= 0 else RED for v in df["MACD_histogram"].fillna(0)]
        fig2.add_trace(go.Bar(
            x=df.index, y=df["MACD_histogram"], name="Histogram",
            marker_color=hist_colors, opacity=0.7,
        ), row=2, col=1)

    # RSI
    if df["RSI"].notna().any():
        fig2.add_trace(go.Scatter(
            x=df.index, y=df["RSI"], name="RSI",
            line=dict(color="#a29bfe", width=1.5),
        ), row=3, col=1)
        fig2.add_hline(y=70, line_dash="dash", line_color=RED, opacity=0.5, row=3, col=1)
        fig2.add_hline(y=30, line_dash="dash", line_color=TEAL, opacity=0.5, row=3, col=1)
        fig2.update_yaxes(range=[0, 100], row=3, col=1)

    # Stochastic
    if df["Stoch_K"].notna().any():
        fig2.add_trace(go.Scatter(
            x=df.index, y=df["Stoch_K"], name="%K",
            line=dict(color=TEAL, width=1.5),
        ), row=4, col=1)
        fig2.add_trace(go.Scatter(
            x=df.index, y=df["Stoch_D"], name="%D",
            line=dict(color="#f7b731", width=1.5),
        ), row=4, col=1)
        fig2.add_hline(y=80, line_dash="dash", line_color=RED, opacity=0.5, row=4, col=1)
        fig2.add_hline(y=20, line_dash="dash", line_color=TEAL, opacity=0.5, row=4, col=1)
        fig2.update_yaxes(range=[0, 100], row=4, col=1)

    fig2.update_layout(
        **CHART_LAYOUT,
        height=900,
        showlegend=True,
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Signal Summary table
    st.markdown("### Signal Summary")
    signals_list = interpret_signals(df)

    if signals_list:
        pill_map = {"green": "pill-green", "red": "pill-red", "gray": "pill-gray"}
        rows_html = ""
        for s in signals_list:
            pill_cls = pill_map.get(s["color"], "pill-gray")
            rows_html += f"""
            <tr>
                <td style="padding:8px 12px;font-weight:600;">{s['Indicator']}</td>
                <td style="padding:8px 12px;font-family:monospace;">{s['Value']}</td>
                <td style="padding:8px 12px;"><span class="{pill_cls}">{s['Signal']}</span></td>
                <td style="padding:8px 12px;color:#aaa;">{s['Implication']}</td>
            </tr>"""

        table_html = f"""
        <table style="width:100%;border-collapse:collapse;background:#1a1f2e;border-radius:10px;overflow:hidden;">
            <thead>
                <tr style="background:#2a2f3e;color:#aaa;font-size:0.85rem;text-transform:uppercase;">
                    <th style="padding:10px 12px;text-align:left;">Indicator</th>
                    <th style="padding:10px 12px;text-align:left;">Value</th>
                    <th style="padding:10px 12px;text-align:left;">Signal</th>
                    <th style="padding:10px 12px;text-align:left;">Implication</th>
                </tr>
            </thead>
            <tbody>{rows_html}</tbody>
        </table>"""
        st.markdown(table_html, unsafe_allow_html=True)
    else:
        st.info("Not enough data to generate signals.")


# ============================================================
# TAB 3 — PREDICTION
# ============================================================
with tab3:
    if len(df) < 80:
        st.warning(
            "Not enough historical data for prediction. "
            "At least 80 trading days are required — try a longer period (1Y or 2Y)."
        )
    else:
        with st.spinner("Running ML prediction…"):
            pred_result = predict(df, ticker=ticker)

        if pred_result is None:
            st.warning("Prediction could not be generated (insufficient data after feature engineering).")
        else:
            prediction = pred_result["prediction"]
            confidence = pred_result["confidence"]
            accuracy = pred_result["accuracy"]
            feature_imp = pred_result["feature_importance"]
            signals = pred_result["signals"]

            # Prediction badge
            badge_cls = "pred-up" if prediction == "UP" else "pred-down"
            arrow = "▲" if prediction == "UP" else "▼"
            col_a, col_b, col_c = st.columns([1, 2, 1])
            with col_b:
                st.markdown(
                    f"""
                    <div style="text-align:center; margin: 20px 0;">
                        <p style="color:#aaa;font-size:1rem;margin-bottom:8px;">Next-Day Prediction</p>
                        <div class="pred-badge {badge_cls}">{arrow} {prediction}</div>
                        <p style="color:#aaa;font-size:1.1rem;margin-top:12px;">
                            Confidence: <strong style="color:#fafafa;">{confidence*100:.1f}%</strong>
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Gauge + accuracy
            col1, col2 = st.columns([3, 1])
            with col1:
                gauge_color = TEAL if prediction == "UP" else RED
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=confidence * 100,
                    number={"suffix": "%", "font": {"size": 36, "color": "#fafafa"}},
                    gauge={
                        "axis": {"range": [0, 100], "tickcolor": "#555"},
                        "bar": {"color": gauge_color, "thickness": 0.3},
                        "bgcolor": "#1a1f2e",
                        "bordercolor": "#2a2f3e",
                        "steps": [
                            {"range": [0, 50], "color": "rgba(255,75,110,0.1)"},
                            {"range": [50, 100], "color": "rgba(0,212,170,0.1)"},
                        ],
                        "threshold": {
                            "line": {"color": gauge_color, "width": 3},
                            "thickness": 0.75,
                            "value": confidence * 100,
                        },
                    },
                    title={"text": "Prediction Confidence", "font": {"color": "#aaa"}},
                ))
                fig_gauge.update_layout(
                    **CHART_LAYOUT,
                    height=300,
                )
                st.plotly_chart(fig_gauge, use_container_width=True)

            with col2:
                st.markdown("<div style='height:60px'></div>", unsafe_allow_html=True)
                st.metric("Model Accuracy", f"{accuracy*100:.1f}%",
                          help="Accuracy on the last 60 trading days (test set)")
                st.caption("Test set: last 60 trading days")

            # Feature importances
            st.markdown("### Top 10 Feature Importances")
            if feature_imp:
                feat_df = pd.DataFrame(
                    list(feature_imp.items()), columns=["Feature", "Importance"]
                ).sort_values("Importance")

                fig_feat = go.Figure(go.Bar(
                    x=feat_df["Importance"],
                    y=feat_df["Feature"],
                    orientation="h",
                    marker_color=TEAL,
                    marker_line_width=0,
                ))
                fig_feat.update_layout(
                    **CHART_LAYOUT,
                    height=360,
                    xaxis_title="Importance Score",
                    yaxis_title=None,
                )
                st.plotly_chart(fig_feat, use_container_width=True)

            # Indicator signal cards
            st.markdown("### Current Indicator Signals")
            signal_colors = {
                "Oversold": TEAL, "Overbought": RED,
                "Bullish": TEAL, "Bearish": RED,
                "Bullish crossover": TEAL, "Bearish crossover": RED,
                "Near upper band": RED, "Near lower band": TEAL,
                "Price above SMA20": TEAL, "Price below SMA20": RED,
                "Price above SMA50": TEAL, "Price below SMA50": RED,
                "High volatility": RED, "Low volatility": "#aaa",
                "Normal volatility": "#aaa", "Within bands": "#aaa",
                "Neutral": "#aaa",
            }
            card_cols = st.columns(3)
            for idx, (sig_name, sig_val) in enumerate(signals.items()):
                color = signal_colors.get(sig_val, "#aaa")
                with card_cols[idx % 3]:
                    st.markdown(
                        f"""
                        <div style="background:#1a1f2e;border:1px solid #2a2f3e;border-radius:10px;
                                    padding:14px 16px;margin-bottom:10px;border-top:3px solid {color};">
                            <div style="color:#888;font-size:0.8rem;text-transform:uppercase;margin-bottom:4px;">
                                {sig_name}
                            </div>
                            <div style="color:{color};font-weight:700;font-size:1rem;">{sig_val}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            # Disclaimer
            st.markdown(
                """
                <div style="background:rgba(255,75,110,0.08);border:1px solid rgba(255,75,110,0.3);
                            border-radius:8px;padding:12px 16px;margin-top:20px;">
                    ⚠️ <strong>Disclaimer:</strong> This is not financial advice.
                    Model accuracy is not guaranteed. Past performance does not predict future results.
                    Always do your own research before making investment decisions.
                </div>
                """,
                unsafe_allow_html=True,
            )


# ============================================================
# TAB 4 — SENTIMENT
# ============================================================
with tab4:
    if news is None:
        st.error("Failed to fetch news articles.")
    elif len(news) == 0:
        st.warning(f"No recent news found for **{ticker}**.")
    else:
        with st.spinner("Analyzing sentiment…"):
            sent_result = analyze_sentiment(news)

        if sent_result is None:
            st.warning("Could not perform sentiment analysis (no articles).")
        else:
            overall = sent_result["overall_sentiment"]
            avg_score = sent_result["avg_compound_score"]
            breakdown = sent_result["sentiment_breakdown"]
            articles = sent_result["articles"]

            # Overall badge
            badge_cls_map = {"Positive": "sent-pos", "Negative": "sent-neg", "Neutral": "sent-neu"}
            badge_cls = badge_cls_map.get(overall, "sent-neu")
            score_color = TEAL if overall == "Positive" else (RED if overall == "Negative" else "#aaa")

            col_a, col_b, col_c = st.columns([1, 2, 1])
            with col_b:
                st.markdown(
                    f"""
                    <div style="text-align:center; margin: 16px 0;">
                        <p style="color:#aaa;font-size:1rem;margin-bottom:8px;">Overall News Sentiment</p>
                        <div class="sent-badge {badge_cls}">{overall}</div>
                        <p style="color:{score_color};font-size:1.1rem;margin-top:10px;">
                            Avg. Compound Score: <strong>{avg_score:+.3f}</strong>
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Donut chart
            col1, col2 = st.columns([1, 1])
            with col1:
                st.markdown("### Sentiment Breakdown")
                labels = list(breakdown.keys())
                values = list(breakdown.values())
                donut_colors = [TEAL, "#aaaaaa", RED]

                fig_donut = go.Figure(go.Pie(
                    labels=labels,
                    values=values,
                    hole=0.55,
                    marker=dict(colors=donut_colors, line=dict(color="#0e1117", width=2)),
                    textinfo="label+percent",
                    textfont=dict(color="#fafafa"),
                ))
                fig_donut.update_layout(
                    **CHART_LAYOUT,
                    height=320,
                    showlegend=False,
                    annotations=[dict(
                        text=f"{sum(values)}<br>articles",
                        x=0.5, y=0.5, font_size=16,
                        font_color="#fafafa", showarrow=False,
                    )],
                )
                st.plotly_chart(fig_donut, use_container_width=True)

            with col2:
                st.markdown("### Counts")
                for label, cnt in breakdown.items():
                    clr = TEAL if label == "Positive" else (RED if label == "Negative" else "#aaa")
                    st.markdown(
                        f"""
                        <div style="background:#1a1f2e;border-radius:8px;padding:12px 16px;
                                    margin-bottom:8px;border-left:4px solid {clr};">
                            <span style="color:{clr};font-weight:700;">{label}</span>
                            <span style="float:right;color:#fafafa;font-size:1.3rem;font-weight:800;">{cnt}</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            # News feed
            st.markdown("### Recent News")
            label_cls = {"Positive": "news-pos", "Negative": "news-neg", "Neutral": "news-neu"}
            label_pill = {"Positive": "pill-green", "Negative": "pill-red", "Neutral": "pill-gray"}

            for art in articles:
                row_cls = label_cls.get(art["label"], "news-neu")
                pill_cls = label_pill.get(art["label"], "pill-gray")
                pub_date = art["published_at"]
                date_str = pub_date.strftime("%b %d, %Y %H:%M UTC") if pub_date else "Date unknown"

                st.markdown(
                    f"""
                    <div class="news-row {row_cls}">
                        <a href="{art['link']}" target="_blank"
                           style="color:#fafafa;font-weight:600;font-size:0.95rem;text-decoration:none;">
                            {art['title']}
                        </a>
                        <div style="margin-top:6px;color:#888;font-size:0.8rem;">
                            {art['publisher']} &nbsp;·&nbsp; {date_str}
                            &nbsp;&nbsp;
                            <span class="{pill_cls}">{art['label']} {art['compound']:+.2f}</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
