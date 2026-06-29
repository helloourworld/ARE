import sys
import pandas as pd
from pathlib import Path

try:
    from ..data_pipeline import data_cache
except ImportError:
    try:
        from data_pipeline import data_cache
    except ImportError:
        package_root = Path(__file__).resolve().parent.parent
        if str(package_root) not in sys.path:
            sys.path.insert(0, str(package_root))
        from data_pipeline import data_cache


class DummyYF:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def download(self, ticker, period=None, interval=None, start=None, prepost=True, progress=False):
        self.calls.append({"ticker": ticker, "period": period,
                          "interval": interval, "start": start})
        return self.payload


def test_get_data_persistent_uses_existing_cache(tmp_path, monkeypatch):
    cache_dir = tmp_path / "data"
    cache_dir.mkdir()

    existing = pd.DataFrame(
        {"Open": [10.0], "High": [11.0], "Low": [
            9.0], "Close": [10.5], "Volume": [1000]},
        index=pd.to_datetime(["2024-01-01 09:30:00"]),
    )
    existing.to_csv(cache_dir / "cache_AAPL_1d.csv")

    monkeypatch.setattr(data_cache, "DATA_DIR", cache_dir)
    monkeypatch.setattr(data_cache, "REPO_ROOT", tmp_path)

    monkeypatch.setattr(data_cache, "yf", DummyYF(pd.DataFrame()))

    result = data_cache.get_data_persistent("AAPL", interval="1d", period="2y")

    assert not result.empty
    assert list(result.columns)[:5] == [
        "Open", "High", "Low", "Close", "Volume"]
