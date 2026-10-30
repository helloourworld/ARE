import pandas as pd
import numpy as np

from risk_modeling.market_breadth import calculate_breadth, breadth_signal


def test_three_day_ad_confirmation_requires_persistence():
    dates = pd.date_range("2024-01-01", periods=3, freq="D")
    close = pd.DataFrame(
        {
            "AAA": [100.0, 102.0, 104.0],
            "BBB": [100.0, 101.0, 103.0],
        },
        index=dates,
    )

    breadth = calculate_breadth(close)
    latest = breadth.iloc[-1]
    assert np.isinf(latest["A/D Ratio"])
    assert pd.isna(latest["3-Day A/D Average"])
    assert latest["3-Day Positive A/D Count"] == 2.0
    assert bool(latest["A/D Positive Confirmation"]) is True


def test_zero_decliners_is_positive_infinity():
    close = pd.DataFrame(
        {"AAA": [100.0, 101.0], "BBB": [100.0, 102.0]},
        index=pd.date_range("2024-01-01", periods=2, freq="D"),
    )
    breadth = calculate_breadth(close)

    assert breadth["A/D Ratio"].iloc[-1] == float("inf")
    assert bool(breadth["A/D Positive Confirmation"].iloc[-1]) is False


def test_three_day_negative_ad_confirmation():
    close = pd.DataFrame(
        {
            "AAA": [100.0, 98.0, 96.0, 94.0],
            "BBB": [100.0, 99.0, 97.0, 95.0],
        },
        index=pd.date_range("2024-01-01", periods=4, freq="D"),
    )
    breadth = calculate_breadth(close)
    latest = breadth.iloc[-1]

    assert latest["A/D Ratio"] == 0.0
    assert latest["3-Day A/D Average"] == 0.0
    assert latest["3-Day Negative A/D Count"] == 3.0
    assert bool(latest["A/D Negative Confirmation"]) is True


def test_confirmed_positive_ad_adds_one_point():
    latest = {
        "% Above 50DMA": 0.50,
        "% Above 200DMA": 0.50,
        "A/D Ratio": 1.20,
        "A/D Positive Confirmation": True,
        "A/D Negative Confirmation": False,
        "3-Day A/D Average": 1.60,
        "3-Day Positive A/D Count": 2,
        "52W New Highs": 1,
        "52W New Lows": 1,
    }

    _, _, score, details = breadth_signal(latest)

    assert score == 1
    assert any("3-day A/D confirmed bullish" in detail for detail in details)


def test_confirmed_negative_ad_adds_negative_one_point():
    latest = {
        "% Above 50DMA": 0.50,
        "% Above 200DMA": 0.50,
        "A/D Ratio": 0.50,
        "A/D Positive Confirmation": False,
        "A/D Negative Confirmation": True,
        "3-Day A/D Average": 0.50,
        "3-Day Positive A/D Count": 0,
        "3-Day Negative A/D Count": 3,
        "52W New Highs": 1,
        "52W New Lows": 1,
    }

    _, _, score, details = breadth_signal(latest)

    assert score == -1
    assert any("negative days 3 of 3" in detail for detail in details)
