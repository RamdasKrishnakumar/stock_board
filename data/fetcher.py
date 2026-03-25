"""
Data fetching layer — wraps yfinance with Streamlit caching.
All public functions return None on failure and call st.error() internally.
"""

import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timezone


# ---------------------------------------------------------------------------
# Price history
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300)  # 5-minute TTL
def fetch_price_history(ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame | None:
    """
    Fetch OHLCV price history for *ticker*.

    Parameters
    ----------
    ticker   : stock symbol, e.g. 'AAPL'
    period   : yfinance period string — '1mo','3mo','6mo','1y','2y','5y'
    interval : yfinance interval string — '1d','1h', etc.

    Returns
    -------
    DataFrame with columns [Open, High, Low, Close, Volume] indexed by Date,
    or None on failure.
    """
    try:
        ticker_obj = yf.Ticker(ticker.upper())
        df = ticker_obj.history(period=period, interval=interval, auto_adjust=True)

        if df is None or df.empty:
            st.error(f"No price data found for **{ticker.upper()}**. Check the ticker symbol and try again.")
            return None

        # Keep only OHLCV columns, drop timezone from index for easier handling
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df.index.name = "Date"
        df.dropna(subset=["Close"], inplace=True)
        return df

    except Exception as exc:
        st.error(f"Failed to fetch price data for **{ticker.upper()}**: {exc}")
        return None


# ---------------------------------------------------------------------------
# Company info
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)  # 1-hour TTL
def fetch_company_info(ticker: str) -> dict | None:
    """
    Fetch fundamental company information.

    Returns a dict with keys:
        name, sector, market_cap, pe_ratio, week_52_high, week_52_low,
        description, currency, current_price, previous_close, day_change_pct
    or None on failure.
    """
    try:
        ticker_obj = yf.Ticker(ticker.upper())
        info = ticker_obj.info

        if not info or info.get("regularMarketPrice") is None and info.get("currentPrice") is None:
            # Try fast_info as fallback
            fast = ticker_obj.fast_info
            current_price = getattr(fast, "last_price", None)
            if current_price is None:
                st.error(f"Could not retrieve company information for **{ticker.upper()}**.")
                return None

        current_price = info.get("currentPrice") or info.get("regularMarketPrice") or info.get("navPrice")
        previous_close = info.get("previousClose") or info.get("regularMarketPreviousClose")

        day_change_pct = None
        if current_price and previous_close and previous_close != 0:
            day_change_pct = ((current_price - previous_close) / previous_close) * 100

        market_cap = info.get("marketCap")
        if market_cap:
            if market_cap >= 1_000_000_000_000:
                market_cap_str = f"${market_cap / 1_000_000_000_000:.2f}T"
            elif market_cap >= 1_000_000_000:
                market_cap_str = f"${market_cap / 1_000_000_000:.2f}B"
            elif market_cap >= 1_000_000:
                market_cap_str = f"${market_cap / 1_000_000:.2f}M"
            else:
                market_cap_str = f"${market_cap:,.0f}"
        else:
            market_cap_str = "N/A"

        return {
            "name": info.get("longName") or info.get("shortName") or ticker.upper(),
            "sector": info.get("sector") or "N/A",
            "industry": info.get("industry") or "N/A",
            "market_cap": market_cap_str,
            "pe_ratio": info.get("trailingPE") or info.get("forwardPE"),
            "week_52_high": info.get("fiftyTwoWeekHigh"),
            "week_52_low": info.get("fiftyTwoWeekLow"),
            "description": info.get("longBusinessSummary") or "No description available.",
            "currency": info.get("currency") or "USD",
            "current_price": current_price,
            "previous_close": previous_close,
            "day_change_pct": day_change_pct,
        }

    except Exception as exc:
        st.error(f"Failed to fetch company info for **{ticker.upper()}**: {exc}")
        return None


# ---------------------------------------------------------------------------
# News
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)  # 1-hour TTL
def fetch_news(ticker: str, limit: int = 20) -> list[dict] | None:
    """
    Fetch recent news articles for *ticker*.

    Each returned dict contains:
        title, publisher, link, published_at (datetime or None)
    Returns None on failure, empty list if no news found.
    """
    try:
        ticker_obj = yf.Ticker(ticker.upper())
        raw_news = ticker_obj.news

        if raw_news is None:
            return []

        articles = []
        for item in raw_news[:limit]:
            # yfinance returns providerPublishTime as a Unix timestamp
            pub_ts = item.get("providerPublishTime")
            if pub_ts:
                try:
                    published_at = datetime.fromtimestamp(pub_ts, tz=timezone.utc).replace(tzinfo=None)
                except Exception:
                    published_at = None
            else:
                published_at = None

            # Handle both old and new yfinance news schema
            content = item.get("content", {})
            if isinstance(content, dict):
                title = content.get("title") or item.get("title", "No title")
                link = (content.get("canonicalUrl", {}) or {}).get("url") or item.get("link", "#")
                provider = (content.get("provider", {}) or {}).get("displayName") or item.get("publisher", "Unknown")
                pub_date = content.get("pubDate")
                if pub_date:
                    try:
                        published_at = datetime.fromisoformat(pub_date.replace("Z", "+00:00")).replace(tzinfo=None)
                    except Exception:
                        pass
            else:
                title = item.get("title", "No title")
                link = item.get("link", "#")
                provider = item.get("publisher", "Unknown")

            articles.append({
                "title": title,
                "publisher": provider,
                "link": link,
                "published_at": published_at,
            })

        return articles

    except Exception as exc:
        st.error(f"Failed to fetch news for **{ticker.upper()}**: {exc}")
        return None
