import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
import pandas as pd
import yfinance as yf

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def _get_cache_path(file_name: str) -> Path:
    target_path = DATA_DIR / file_name
    legacy_paths = [Path(file_name), REPO_ROOT / file_name]
    for legacy_path in legacy_paths:
        if legacy_path.exists() and legacy_path != target_path:
            try:
                legacy_path.replace(target_path)
            except OSError:
                pass
    return target_path


def _normalize_yf_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df


def get_data_persistent(ticker, interval="1d", period="2y", force_refresh=False):
    """Load OHLCV data from a local cache first, then refresh from Yahoo Finance as needed."""
    safe_ticker = str(ticker).replace("/", "_").replace("=", "_")
    file_path = _get_cache_path(f"cache_{safe_ticker}_{interval}.csv")
    now = datetime.now(ZoneInfo("America/Halifax"))

    try:
        if file_path.exists() and not force_refresh:
            local_df = pd.read_csv(file_path, index_col=0, parse_dates=True)
            local_df = _normalize_yf_df(local_df)

            if local_df.empty:
                return local_df

            last_ts = local_df.index[-2] # because the latest volume bar may be incomplete, we use the second to last timestamp for comparison
            wait_time = timedelta(minutes=2) if interval == "1m" else timedelta(hours=12)
            if now - last_ts < wait_time:
                return local_df

            start_date = max(last_ts, now - timedelta(days=7)) if interval == "1m" else last_ts
            if hasattr(start_date, "to_pydatetime"):
                start_date = start_date.to_pydatetime()

            new_data = yf.download(
                ticker,
                start=start_date,
                interval=interval,
                prepost=True,
                progress=False,
            )
            if not new_data.empty:
                new_data = new_data[1:]  # skip the first row to avoid duplicates
                new_data = _normalize_yf_df(new_data)
                combined = pd.concat([local_df, new_data])
                combined = combined[~combined.index.duplicated(keep="last")].sort_index()
                combined.to_csv(file_path)
                return combined
            return local_df
        p = "7d" if interval == "1m" else period
        print(f"initializing download for {ticker} with interval {interval} and period {p}")
        df = yf.download(ticker, period=p, interval=interval, prepost=True, progress=True)
        if df is None or df.empty:
            return pd.DataFrame()
        df = _normalize_yf_df(df)
        df.to_csv(file_path)
        return df
    except Exception as e:
        print("Download/cache error:", repr(e))
        return pd.DataFrame()
    
if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else "SPY"
    interval = sys.argv[2] if len(sys.argv) > 2 else "1m"
    period = sys.argv[3] if len(sys.argv) > 3 else "7d"
    force_refresh = bool(int(sys.argv[4])) if len(sys.argv) > 4 else False

    df = get_data_persistent(ticker, interval, period, force_refresh)
    print(df.tail())    
