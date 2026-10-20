"""Fetch Yahoo Finance news and calculate FinBERT sentiment features."""

from __future__ import annotations

import logging
import math
from functools import lru_cache
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_FINBERT_MODEL = "ProsusAI/finbert"
NEWS_COLUMNS = ("published_at", "ticker", "title", "summary", "provider", "url")


class MarketNewsError(RuntimeError):
    """Raised when a news provider fails and no usable fallback is available."""


class _TextOnlyHTMLParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def _plain_text(value: str) -> str:
    parser = _TextOnlyHTMLParser()
    parser.feed(value)
    return " ".join(" ".join(parser.parts).split())


def _article_content(article: dict[str, Any]) -> dict[str, Any]:
    content = article.get("content")
    return content if isinstance(content, dict) else article


def _nested_url(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("url") or value.get("displayUrl") or "")
    return str(value or "")


def _published_at(article: dict[str, Any], content: dict[str, Any]) -> pd.Timestamp:
    value = (
        content.get("pubDate")
        or article.get("pubDate")
        or content.get("providerPublishTime")
        or article.get("providerPublishTime")
        or content.get("published_at")
        or article.get("published_at")
    )
    if value is None:
        return pd.NaT
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return pd.to_datetime(value, unit="s", utc=True, errors="coerce")
    return pd.to_datetime(value, utc=True, errors="coerce")


def _fetch_google_news_rss(ticker: str, limit: int) -> list[dict[str, Any]]:
    query = urlencode({
        "q": f'"{ticker}" stock',
        "hl": "en-US",
        "gl": "US",
        "ceid": "US:en",
    })
    request = Request(
        f"https://news.google.com/rss/search?{query}",
        headers={"User-Agent": "Mozilla/5.0 ARE market-news reader"},
    )
    with urlopen(request, timeout=15) as response:
        payload = response.read(2_000_000)
    root = ET.fromstring(payload)
    articles = []
    for item in root.findall("./channel/item")[:limit]:
        title = (item.findtext("title") or "").strip()
        source = (item.findtext("source") or "").strip()
        if not source and " - " in title:
            title, source = title.rsplit(" - ", 1)
        articles.append({
            "title": title,
            "summary": _plain_text(item.findtext("description") or ""),
            "pubDate": item.findtext("pubDate"),
            "publisher": source,
            "link": item.findtext("link"),
            "relatedTickers": [ticker],
        })
    return articles


def normalize_news_articles(
    articles: list[dict[str, Any]],
    ticker: str,
) -> pd.DataFrame:
    """Normalize Google News RSS and Yahoo Finance news article formats."""
    symbol = ticker.strip().upper()
    rows = []
    for article in articles:
        if not isinstance(article, dict):
            continue
        content = _article_content(article)
        title = str(content.get("title") or article.get("title") or "").strip()
        if not title:
            continue

        related = content.get("relatedTickers") or article.get("relatedTickers") or []
        related_symbols = {
            str(item.get("ticker", "") if isinstance(item, dict) else item).upper()
            for item in related
        }
        if related_symbols and symbol not in related_symbols:
            continue

        provider = (
            content.get("provider")
            or article.get("provider")
            or content.get("publisher")
            or article.get("publisher")
            or {}
        )
        provider_name = (
            provider.get("displayName") or provider.get("name") or ""
            if isinstance(provider, dict)
            else str(provider)
        )
        canonical_url = content.get("canonicalUrl") or article.get("canonicalUrl")
        clickthrough_url = content.get("clickThroughUrl") or article.get("clickThroughUrl")
        url = _nested_url(canonical_url) or _nested_url(clickthrough_url)
        if not url:
            url = str(content.get("link") or article.get("link") or "")

        rows.append({
            "published_at": _published_at(article, content),
            "ticker": symbol,
            "title": title,
            "summary": str(
                content.get("summary")
                or content.get("description")
                or article.get("summary")
                or article.get("description")
                or ""
            ).strip(),
            "provider": provider_name,
            "url": url,
        })

    frame = pd.DataFrame(rows, columns=NEWS_COLUMNS)
    if frame.empty:
        return frame
    frame["_dedupe_key"] = np.where(
        frame["url"].ne(""),
        "url:" + frame["url"],
        "title:" + frame["title"].str.casefold() + "|"
        + frame["published_at"].astype(str),
    )
    frame = frame.drop_duplicates(subset=["_dedupe_key"], keep="first").drop(
        columns="_dedupe_key"
    )
    return frame.sort_values("published_at", ascending=False, na_position="last").reset_index(drop=True)


def fetch_market_news(
    ticker: str,
    lookback_hours: int = 72,
    limit: int = 50,
    as_of: Any = None,
) -> pd.DataFrame:
    """Fetch and time-filter ticker news from Yahoo Finance.

    Empty or unavailable Yahoo news falls back to Google News RSS, which is for
    personal, non-commercial feed-reader use.
    """
    symbol = ticker.strip().upper()
    if not symbol:
        raise ValueError("Ticker must not be empty.")
    if lookback_hours <= 0:
        raise ValueError("lookback_hours must be positive.")
    if limit <= 0:
        raise ValueError("limit must be positive.")

    yahoo_error = None
    import yfinance as yf

    try:
        raw_articles = yf.Ticker(symbol).get_news(count=min(limit, 100), tab="all")
    except (yf.exceptions.YFException, OSError) as exc:
        yahoo_error = exc
        logger.warning(
            "Yahoo Finance news unavailable; trying Google News RSS fallback | ticker=%s | error_type=%s",
            symbol,
            type(exc).__name__,
        )
        raw_articles = []
    if not raw_articles:
        try:
            raw_articles = _fetch_google_news_rss(symbol, limit)
        except (OSError, ET.ParseError) as exc:
            failure_detail = (
                f"Yahoo Finance failed ({type(yahoo_error).__name__}: {yahoo_error}); "
                if yahoo_error is not None
                else "Yahoo Finance returned no headlines; "
            )
            raise MarketNewsError(
                f"{failure_detail}Google News RSS fallback failed for {symbol} "
                f"({type(exc).__name__}: {exc})."
            ) from exc
    news = normalize_news_articles(raw_articles, symbol)
    if news.empty:
        return news

    now = pd.Timestamp.now(tz="UTC") if as_of is None else pd.Timestamp(as_of)
    now = now.tz_localize("UTC") if now.tzinfo is None else now.tz_convert("UTC")
    cutoff = now - pd.Timedelta(hours=lookback_hours)
    missing_dates = int(news["published_at"].isna().sum())
    if missing_dates:
        logger.warning(
            "Ignoring news without a valid publication time | ticker=%s | count=%s",
            symbol,
            missing_dates,
        )
    news = news.loc[news["published_at"].notna()]
    news = news.loc[(news["published_at"] <= now) & (news["published_at"] >= cutoff)]
    return news.head(limit).reset_index(drop=True)


@lru_cache(maxsize=2)
def _load_finbert(model_name: str):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return tokenizer, model, device


def score_news_with_finbert(
    news: pd.DataFrame,
    model_name: str = DEFAULT_FINBERT_MODEL,
    batch_size: int = 16,
) -> pd.DataFrame:
    """Score each headline with FinBERT; score is P(positive) - P(negative)."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    scored = news.copy()
    scored["text"] = (
        scored["title"].fillna("").astype(str)
        + ". "
        + scored["summary"].fillna("").astype(str)
    ).str.strip()
    if scored.empty:
        scored["sentiment_score"] = pd.Series(dtype=float)
        scored["sentiment_confidence"] = pd.Series(dtype=float)
        scored["sentiment_label"] = pd.Series(dtype=str)
        return scored

    import torch

    tokenizer, model, device = _load_finbert(model_name)
    label_names = {
        int(index): str(label).lower()
        for index, label in model.config.id2label.items()
    }
    positive_ids = [index for index, label in label_names.items() if "positive" in label]
    negative_ids = [index for index, label in label_names.items() if "negative" in label]
    if len(positive_ids) != 1 or len(negative_ids) != 1:
        raise ValueError(
            f"Model {model_name!r} must expose one positive and one negative class; "
            f"found labels {label_names}."
        )

    all_probabilities = []
    texts = scored["text"].tolist()
    for start in range(0, len(texts), batch_size):
        inputs = tokenizer(
            texts[start : start + batch_size],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.inference_mode():
            logits = model(**inputs).logits
            probabilities = torch.softmax(logits, dim=-1).cpu().numpy()
        all_probabilities.append(probabilities)

    probabilities = np.concatenate(all_probabilities, axis=0)
    positive = probabilities[:, positive_ids[0]]
    negative = probabilities[:, negative_ids[0]]
    scores = positive - negative
    scored["sentiment_score"] = scores
    scored["sentiment_confidence"] = probabilities.max(axis=1)
    scored["sentiment_label"] = np.select(
        [scores >= 0.15, scores <= -0.15],
        ["Positive", "Negative"],
        default="Neutral",
    )
    return scored


def compute_news_indicators(
    scored_news: pd.DataFrame,
    as_of: Any = None,
    half_life_hours: float = 24.0,
) -> dict[str, Any]:
    """Aggregate article scores into recency-weighted, interpretable features."""
    if not math.isfinite(half_life_hours) or half_life_hours <= 0:
        raise ValueError("half_life_hours must be a finite positive number.")
    now = pd.Timestamp.now(tz="UTC") if as_of is None else pd.Timestamp(as_of)
    now = now.tz_localize("UTC") if now.tzinfo is None else now.tz_convert("UTC")

    empty = {
        "news_count": 0,
        "news_sentiment": float("nan"),
        "sentiment_breadth": float("nan"),
        "sentiment_confidence": float("nan"),
        "effective_news_count": 0.0,
        "positive_share": float("nan"),
        "neutral_share": float("nan"),
        "negative_share": float("nan"),
        "signal": "Insufficient news",
        "signal_strength": float("nan"),
    }
    required = {"published_at", "sentiment_score", "sentiment_confidence"}
    if scored_news.empty:
        return empty
    missing = required - set(scored_news.columns)
    if missing:
        raise ValueError(f"Scored news is missing required columns: {', '.join(sorted(missing))}")

    data = scored_news.copy()
    data["published_at"] = pd.to_datetime(data["published_at"], utc=True, errors="coerce")
    data["sentiment_score"] = pd.to_numeric(data["sentiment_score"], errors="coerce")
    data["sentiment_confidence"] = pd.to_numeric(data["sentiment_confidence"], errors="coerce")
    data = data.dropna(subset=["published_at", "sentiment_score", "sentiment_confidence"])
    data = data.loc[
        data["published_at"].le(now)
        & data["sentiment_score"].between(-1, 1)
        & data["sentiment_confidence"].between(0, 1)
    ].copy()
    if data.empty:
        return empty

    ages = (now - data["published_at"]).dt.total_seconds().clip(lower=0) / 3600
    weights = np.exp(-math.log(2) * ages / half_life_hours)
    weighted_score = float(np.average(data["sentiment_score"], weights=weights))
    labels = np.select(
        [data["sentiment_score"] >= 0.15, data["sentiment_score"] <= -0.15],
        ["Positive", "Negative"],
        default="Neutral",
    )
    total = len(data)
    positive_share = float(np.mean(labels == "Positive"))
    neutral_share = float(np.mean(labels == "Neutral"))
    negative_share = float(np.mean(labels == "Negative"))
    signal = (
        "Insufficient news"
        if total < 3
        else "Positive sentiment"
        if weighted_score >= 0.15
        else "Negative sentiment"
        if weighted_score <= -0.15
        else "Neutral sentiment"
    )
    return {
        "news_count": total,
        "news_sentiment": weighted_score,
        "sentiment_breadth": positive_share - negative_share,
        "sentiment_confidence": float(
            np.average(data["sentiment_confidence"], weights=weights)
        ),
        "effective_news_count": float(weights.sum()),
        "positive_share": positive_share,
        "neutral_share": neutral_share,
        "negative_share": negative_share,
        "signal": signal,
        "signal_strength": (
            weighted_score * min(1.0, total / 10.0)
            if total >= 3
            else float("nan")
        ),
    }


def add_decayed_sentiment_trend(
    scored_news: pd.DataFrame,
    half_life_hours: float = 24.0,
) -> pd.DataFrame:
    """Add a point-in-time decayed sentiment series for charting."""
    if not math.isfinite(half_life_hours) or half_life_hours <= 0:
        raise ValueError("half_life_hours must be a finite positive number.")
    result = scored_news.copy()
    result["sentiment_trend"] = pd.Series(np.nan, index=result.index, dtype=float)
    if result.empty:
        return result

    required = {"published_at", "sentiment_score"}
    missing = required - set(result.columns)
    if missing:
        raise ValueError(f"Scored news is missing required columns: {', '.join(sorted(missing))}")

    result["published_at"] = pd.to_datetime(result["published_at"], utc=True, errors="coerce")
    result["sentiment_score"] = pd.to_numeric(result["sentiment_score"], errors="coerce")
    result = result.sort_values("published_at")
    valid = result.dropna(subset=["published_at", "sentiment_score"])
    timestamps = valid["published_at"]
    scores = valid["sentiment_score"].to_numpy(dtype=float)
    trend = []
    for index, current_time in enumerate(timestamps):
        ages = (current_time - timestamps.iloc[: index + 1]).dt.total_seconds().to_numpy() / 3600
        weights = np.exp(-math.log(2) * ages / half_life_hours)
        trend.append(float(np.average(scores[: index + 1], weights=weights)))
    result.loc[valid.index, "sentiment_trend"] = trend
    return result.sort_values("published_at", ascending=False, na_position="last").reset_index(drop=True)


def analyze_market_news(
    ticker: str,
    lookback_hours: int = 72,
    limit: int = 50,
    model_name: str = DEFAULT_FINBERT_MODEL,
    as_of: Any = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fetch recent ticker news, score with FinBERT, and return indicators."""
    news = fetch_market_news(
        ticker,
        lookback_hours=lookback_hours,
        limit=limit,
        as_of=as_of,
    )
    scored = score_news_with_finbert(news, model_name=model_name)
    scored = add_decayed_sentiment_trend(scored)
    return scored, compute_news_indicators(scored, as_of=as_of)