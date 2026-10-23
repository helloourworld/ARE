from types import SimpleNamespace

import pandas as pd
import pytest
import torch

from alpha_research.market_news import (
    add_decayed_sentiment_trend,
    compute_news_indicators,
    fetch_market_news,
    normalize_news_articles,
    score_news_with_finbert,
)


def test_normalize_yahoo_news_supports_nested_and_legacy_articles():
    articles = [
        {
            "content": {
                "title": "Company raises guidance",
                "summary": "Management reported stronger demand.",
                "pubDate": "2026-10-20T14:00:00Z",
                "provider": {"displayName": "Example News"},
                "canonicalUrl": {"url": "https://example.test/story"},
                "relatedTickers": [{"ticker": "AMZN"}],
            }
        },
        {
            "title": "Legacy headline",
            "summary": "A legacy-format article.",
            "providerPublishTime": 1_792_510_000,
            "publisher": "Legacy News",
            "link": "https://example.test/legacy",
            "relatedTickers": ["AMZN"],
        },
        {
            "title": "Unrelated symbol story",
            "pubDate": "2026-10-20T14:00:00Z",
            "relatedTickers": ["MSFT"],
        },
    ]

    news = normalize_news_articles(articles, "amzn")

    assert news["ticker"].unique().tolist() == ["AMZN"]
    assert set(news["title"]) == {"Company raises guidance", "Legacy headline"}
    current = news.loc[news["title"] == "Company raises guidance"].iloc[0]
    legacy = news.loc[news["title"] == "Legacy headline"].iloc[0]
    assert current["provider"] == "Example News"
    assert current["url"] == "https://example.test/story"
    assert legacy["provider"] == "Legacy News"
    assert news["published_at"].dt.tz is not None


def test_normalize_yahoo_news_deduplicates_with_and_without_urls():
    articles = [
        {"title": "Same story", "pubDate": "2026-10-20T14:00:00Z", "link": "https://a.test"},
        {"title": "Same story", "pubDate": "2026-10-20T14:00:00Z", "link": "https://a.test"},
        {"title": "No URL", "pubDate": "2026-10-20T14:01:00Z"},
        {"title": "No URL", "pubDate": "2026-10-20T14:01:00Z"},
    ]

    news = normalize_news_articles(articles, "SPY")

    assert len(news) == 2


def test_normalize_yahoo_news_handles_millisecond_timestamps_and_missing_provider():
    articles = [
        {
            "title": "Millisecond timestamp story",
            "providerPublishTime": 1_726_000_000_000,
            "relatedTickers": ["MSFT"],
        },
        {"title": "Missing provider story", "pubDate": "2026-10-20T14:00:00Z", "provider": None},
    ]

    news = normalize_news_articles(articles, "msft")

    assert news["provider"].tolist() == ["", ""]
    assert news["published_at"].dt.tz is not None


def test_fetch_market_news_filters_by_lookback_and_future_date(monkeypatch):
    as_of = pd.Timestamp("2026-10-20T16:00:00Z")
    articles = [
        {
            "title": "Recent story",
            "pubDate": (as_of - pd.Timedelta(hours=1)).isoformat(),
            "link": "https://example.test/recent",
        },
        {
            "title": "Old story",
            "pubDate": (as_of - pd.Timedelta(days=5)).isoformat(),
            "link": "https://example.test/old",
        },
        {
            "title": "Future story",
            "pubDate": (as_of + pd.Timedelta(hours=1)).isoformat(),
            "link": "https://example.test/future",
        },
    ]
    monkeypatch.setattr(
        "alpha_research.market_news._fetch_google_news_rss",
        lambda ticker, limit: articles,
    )

    class EmptyYahooTicker:
        def get_news(self, count, tab):
            return []

    monkeypatch.setattr("yfinance.Ticker", lambda ticker: EmptyYahooTicker())

    news = fetch_market_news("SPY", lookback_hours=72, as_of=as_of)

    assert news["title"].tolist() == ["Recent story"]


def test_yahoo_news_is_primary_and_google_rss_is_fallback(monkeypatch):
    as_of = pd.Timestamp("2026-10-20T16:00:00Z")
    yahoo_articles = [{
        "title": "Yahoo headline",
        "pubDate": (as_of - pd.Timedelta(hours=1)).isoformat(),
        "link": "https://example.test/yahoo",
    }]

    class YahooTicker:
        def get_news(self, count, tab):
            assert count == 50
            assert tab == "all"
            return yahoo_articles

    monkeypatch.setattr("yfinance.Ticker", lambda ticker: YahooTicker())
    monkeypatch.setattr(
        "alpha_research.market_news._fetch_google_news_rss",
        lambda ticker, limit: pytest.fail("Google RSS should only be used as fallback"),
    )

    news = fetch_market_news("SPY", as_of=as_of)

    assert news["title"].tolist() == ["Yahoo headline"]


def test_finbert_scoring_uses_model_label_names_and_batches(monkeypatch):
    class FakeTokenizer:
        def __call__(self, texts, **kwargs):
            ids = [1 if "Negative" in text else 0 for text in texts]
            return {"input_ids": torch.tensor(ids).reshape(-1, 1)}

    class FakeModel:
        config = SimpleNamespace(
            id2label={0: "negative", 1: "neutral", 2: "positive"}
        )

        def __call__(self, input_ids):
            logits = torch.tensor([[0.0, 0.0, 4.0] if value == 0 else [4.0, 0.0, 0.0]
                                  for value in input_ids[:, 0].tolist()])
            return SimpleNamespace(logits=logits)

    monkeypatch.setattr(
        "alpha_research.market_news._load_finbert",
        lambda model_name: (FakeTokenizer(), FakeModel(), torch.device("cpu")),
    )
    news = pd.DataFrame({
        "title": ["Positive headline", "Negative headline"],
        "summary": ["", ""],
    })

    scored = score_news_with_finbert(news, batch_size=1)

    assert scored["sentiment_score"].iloc[0] > 0
    assert scored["sentiment_score"].iloc[1] < 0
    assert scored["sentiment_label"].tolist() == ["Positive", "Negative"]
    assert scored["sentiment_confidence"].between(0, 1).all()


def test_compute_news_indicators_weights_recent_news_and_requires_sample():
    as_of = pd.Timestamp("2026-10-20T16:00:00Z")
    scored_news = pd.DataFrame(
        {
            "published_at": [
                as_of - pd.Timedelta(hours=48),
                as_of - pd.Timedelta(hours=1),
                as_of - pd.Timedelta(minutes=30),
            ],
            "sentiment_score": [-0.8, 0.6, 0.7],
            "sentiment_confidence": [0.9, 0.8, 0.85],
        }
    )

    indicators = compute_news_indicators(scored_news, as_of=as_of)

    assert indicators["news_count"] == 3
    assert indicators["news_sentiment"] > 0
    assert indicators["sentiment_breadth"] == pytest.approx(1 / 3)
    assert indicators["signal"] == "Positive sentiment"
    assert 0 <= indicators["sentiment_confidence"] <= 1
    assert indicators["effective_news_count"] < 3

    insufficient = compute_news_indicators(scored_news.head(2), as_of=as_of)
    assert insufficient["signal"] == "Insufficient news"
    assert pd.isna(insufficient["signal_strength"])


def test_add_decayed_sentiment_trend_is_point_in_time_and_newest_first():
    news = pd.DataFrame(
        {
            "published_at": pd.to_datetime(
                ["2026-10-20T14:00:00Z", "2026-10-20T15:00:00Z"]
            ),
            "sentiment_score": [-0.5, 0.5],
        }
    )

    trend = add_decayed_sentiment_trend(news)

    assert trend["published_at"].is_monotonic_decreasing
    assert trend.iloc[-1]["sentiment_trend"] == pytest.approx(-0.5)
    assert trend.iloc[0]["sentiment_trend"] > 0


def test_compute_news_indicators_returns_explicit_empty_state():
    indicators = compute_news_indicators(pd.DataFrame())

    assert indicators["news_count"] == 0
    assert indicators["signal"] == "Insufficient news"
    assert pd.isna(indicators["news_sentiment"])
