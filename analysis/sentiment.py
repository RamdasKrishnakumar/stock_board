"""
Sentiment analysis on news headlines using VADER.
"""

from datetime import datetime, timedelta

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_analyzer = SentimentIntensityAnalyzer()


def _label(compound: float) -> str:
    if compound >= 0.05:
        return "Positive"
    elif compound <= -0.05:
        return "Negative"
    return "Neutral"


def analyze_sentiment(articles: list[dict]) -> dict | None:
    """
    Run VADER sentiment analysis on a list of news article dicts.

    Each article dict must have: title, publisher, link, published_at (datetime | None)

    Returns
    -------
    dict with:
        overall_sentiment   : "Positive" | "Neutral" | "Negative"
        avg_compound_score  : float
        sentiment_breakdown : {"Positive": int, "Neutral": int, "Negative": int}
        articles            : list of article dicts with added keys:
                                compound, label
        rolling_7d          : list of {"date": date_str, "avg_compound": float}
                              (may be empty if not enough dated articles)
    or None if the articles list is empty/None.
    """
    if not articles:
        return None

    scored = []
    for art in articles:
        title = art.get("title") or ""
        vs = _analyzer.polarity_scores(title)
        compound = vs["compound"]
        scored.append({
            **art,
            "compound": compound,
            "label": _label(compound),
        })

    compounds = [a["compound"] for a in scored]
    avg_compound = sum(compounds) / len(compounds)
    overall = _label(avg_compound)

    breakdown = {"Positive": 0, "Neutral": 0, "Negative": 0}
    for a in scored:
        breakdown[a["label"]] += 1

    rolling_7d = _compute_rolling(scored)

    return {
        "overall_sentiment": overall,
        "avg_compound_score": avg_compound,
        "sentiment_breakdown": breakdown,
        "articles": scored,
        "rolling_7d": rolling_7d,
    }


def _compute_rolling(scored: list[dict], window_days: int = 7) -> list[dict]:
    """
    Compute a rolling *window_days*-day average compound score.
    Returns a list of {"date": str, "avg_compound": float} sorted by date.
    """
    dated = [a for a in scored if a.get("published_at") is not None]
    if len(dated) < 3:
        return []

    try:
        df = pd.DataFrame(dated)[["published_at", "compound"]].copy()
        df["published_at"] = pd.to_datetime(df["published_at"])
        df = df.set_index("published_at").sort_index()

        # Resample to daily mean
        daily = df["compound"].resample("D").mean().dropna()
        if len(daily) < 2:
            return []

        rolling = daily.rolling(window=min(window_days, len(daily))).mean().dropna()

        return [
            {"date": str(date.date()), "avg_compound": float(val)}
            for date, val in rolling.items()
        ]
    except Exception:
        return []
