import json
import os
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd

# FRED API Key. Keep a built-in key as a fallback for local quick-start usage.
API_KEY = os.getenv("FRED_API_KEY", "ef2d97ed23cba762e98908626b8f2a11")

# Core macro indicators that matter for a fundamental screen.
INDICATORS = {
    "Real GDP QoQ change": "A191RL1Q225SBEA",
    "Real GDP YoY change": "A191RL1A225NBEA",
    "After-tax Corporate Profits From Current Production": "CPATAX",
    "Inventories MoM change": "ISRATIO",
    "Retail Sales MoM change": "RSXFS",
    "Real Personal Consumption Expenditures MoM Change": "PCE",
    "Delinquency Rates on Loans at Commercial Banks QoQ Change": "DRALACBN",
    "Monthly change in Nonfarm Payrolls": "PAYEMS",
    "Unemployment rate": "UNRATE",
    "CPI (All Urban Consumers)": "CPIAUCSL",
    "Core CPI": "CPILFESL",
    "Federal Funds Effective Rate": "FEDFUNDS",
    "10-Year Treasury Yield": "DGS10",
}

# Leading indicators that are valid on FRED.
LEADING_INDICATORS = {
    "OECD U.S. Composite Leading Indicator": "USALOLITONOSTSAM",
    "University of Michigan: Consumer Sentiment": "UMCSENT",
    "Smoothed U.S. Recession Probabilities": "RECPROUSM156N",
    "Sahm Rule Recession Indicator": "SAHMCURRENT",
    "All Employees, Manufacturing": "MANEMP",
    "Industrial Production: Manufacturing": "IPMAN",
    "Real Personal Consumption Expenditures": "PCE",
    "Federal Funds Effective Rate": "FEDFUNDS",
}

# A cleaner curated set for the fundamental dashboard.
ECONOMIC_SERIES = {
    "GDP YoY": "A191RL1A225NBEA",
    "Unemployment Rate": "UNRATE",
    "Nonfarm Payrolls": "PAYEMS",
    "CPI": "CPIAUCSL",
    "Core CPI": "CPILFESL",
    "Fed Funds": "FEDFUNDS",
    "2Y Treasury Yield": "DGS2",
    "10Y Treasury Yield": "DGS10",
    "30Y Treasury Yield": "DGS30",
    "Consumer Sentiment": "UMCSENT",
    "Chicago Activity Index": "CFNAI",
    "Chicago Financial Conditions": "NFCI",
    "Recession Probability": "RECPROUSM156N",
    "Industrial Production": "INDPRO",
    "Manufacturing Payrolls": "MANEMP",
    "Sahm Rule": "SAHMCURRENT",
}


def fetch_fred_data(series_id, years=10):
    """Fetch data from the FRED API for a given series ID."""
    end_date = pd.Timestamp.today().normalize()
    start_date = end_date - pd.DateOffset(years=years)
    params = urlencode({
        "series_id": series_id,
        "api_key": API_KEY,
        "file_type": "json",
        "observation_start": start_date.strftime("%Y-%m-%d"),
        "sort_order": "asc",
    })
    url = f"https://api.stlouisfed.org/fred/series/observations?{params}"
    with urlopen(url, timeout=30) as response:
        data = json.loads(response.read().decode("utf-8"))
    return data.get("observations", [])


def _latest_record(observations):
    if not observations:
        return {"date": None, "value": None}
    latest = observations[-1]
    return {"date": latest.get("date"), "value": latest.get("value")}


def get_indicator_history(series_map=None, years=10):
    """Return a DataFrame indexed by date for a set of FRED series."""
    series_map = series_map or ECONOMIC_SERIES
    frames = []
    for name, series_id in series_map.items():
        try:
            observations = fetch_fred_data(series_id, years=years)
        except (HTTPError, URLError, ValueError, KeyError):
            continue

        if not observations:
            continue

        parsed = pd.DataFrame(observations)
        if parsed.empty or "date" not in parsed.columns or "value" not in parsed.columns:
            continue

        cleaned = parsed[["date", "value"]].copy()
        cleaned["date"] = pd.to_datetime(cleaned["date"], errors="coerce")
        cleaned["value"] = pd.to_numeric(cleaned["value"], errors="coerce")
        cleaned = cleaned.dropna(subset=["date", "value"]).sort_values("date")
        if cleaned.empty:
            continue

        cleaned = cleaned.set_index("date").rename(columns={"value": name})
        frames.append(cleaned)

    if not frames:
        return pd.DataFrame()

    combined = frames[0]
    for frame in frames[1:]:
        combined = combined.join(frame, how="outer")
    return combined.sort_index()


def _safe_series_value(series_like):
    if isinstance(series_like, pd.Series):
        values = series_like.dropna()
        if values.empty:
            return None
        return float(values.iloc[-1])
    if isinstance(series_like, (list, tuple)):
        values = [v for v in series_like if pd.notna(v)]
        if not values:
            return None
        return float(values[-1])
    if isinstance(series_like, dict):
        values = [v for v in series_like.values() if pd.notna(v)]
        if not values:
            return None
        return float(values[-1])
    if pd.isna(series_like):
        return None
    return float(series_like)


def compute_macro_health_summary(data):
    """Return a composite macro regime score and label for the dashboard."""
    if isinstance(data, pd.DataFrame):
        latest = {}
        for col in data.columns:
            series = data[col].dropna()
            if not series.empty:
                latest[col] = float(series.iloc[-1])
        if not latest:
            return {"score": 0.0, "label": "No Data", "components": {}}
    elif isinstance(data, dict):
        latest = {}
        for key, value in data.items():
            val = _safe_series_value(value)
            if val is not None:
                latest[key] = val
        if not latest:
            return {"score": 0.0, "label": "No Data", "components": {}}
    else:
        raise TypeError("data must be a pandas DataFrame or dictionary-like mapping")

    def clamp(value):
        return max(0.0, min(100.0, value))

    score_parts = []
    def add_component(name, value, weight):
        if value is None:
            return
        score_parts.append((name, float(value) * float(weight)))

    gdp = _safe_series_value(latest.get("GDP YoY"))
    add_component("GDP", clamp((gdp - 0.5) / 3.5 * 100) if gdp is not None else None, 0.25)

    unemployment = _safe_series_value(latest.get("Unemployment Rate"))
    add_component("Labor", clamp((6.0 - unemployment) / 4.5 * 100) if unemployment is not None else None, 0.18)

    sentiment = _safe_series_value(latest.get("Consumer Sentiment"))
    add_component("Sentiment", clamp((sentiment - 40.0) / 70.0 * 100) if sentiment is not None else None, 0.20)

    chicago = _safe_series_value(latest.get("Chicago Activity Index"))
    add_component("Chicago", clamp((chicago + 1.5) / 3.5 * 100) if chicago is not None else None, 0.15)

    recession = _safe_series_value(latest.get("Recession Probability"))
    add_component("Recession", clamp((1.0 - recession) * 100) if recession is not None else None, 0.12)

    growth = _safe_series_value(latest.get("Industrial Production"))
    add_component("Industry", clamp((growth - 90.0) / 25.0 * 100) if growth is not None else None, 0.10)

    if not score_parts:
        return {"score": 0.0, "label": "No Data", "components": {}}

    weighted_total = sum(v for _, v in score_parts)
    total_weight = sum((0.25, 0.18, 0.20, 0.15, 0.12, 0.10))
    score = clamp(weighted_total / total_weight)

    if score >= 70:
        label = "Expansionary"
    elif score >= 55:
        label = "Neutral"
    elif score >= 40:
        label = "Softening"
    elif score >= 25:
        label = "Cooling"
    else:
        label = "Risky"

    return {
        "score": round(score, 1),
        "label": label,
        "components": {name: round(value, 1) for name, value in score_parts},
    }


def main():
    all_data = {}
    for indicator, series_id in INDICATORS.items():
        print(f"Fetching data for: {indicator}")
        try:
            observations = fetch_fred_data(series_id, years=10)
            all_data[indicator] = _latest_record(observations)
        except (HTTPError, URLError, ValueError, KeyError) as exc:
            print(f"Could not fetch data for {indicator} from FRED. Error: {exc}")
            all_data[indicator] = {"date": None, "value": "Unavailable"}

    for indicator, series_id in LEADING_INDICATORS.items():
        print(f"Fetching data for: {indicator}")
        try:
            observations = fetch_fred_data(series_id, years=10)
            all_data[indicator] = _latest_record(observations)
        except (HTTPError, URLError, ValueError, KeyError) as exc:
            print(f"Could not fetch data for {indicator} from FRED. Error: {exc}")
            all_data[indicator] = {"date": None, "value": "Unavailable"}

    df = pd.DataFrame.from_dict(all_data, orient="index")
    print(df.to_string())


if __name__ == "__main__":
    main()