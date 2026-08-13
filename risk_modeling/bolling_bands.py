import numpy as np
import pandas as pd
from scipy.stats import norm

try:
    import pandas_ta as ta
    _HAS_PANDAS_TA = True
except ImportError:
    ta = None
    _HAS_PANDAS_TA = False

try:
    from .mandelbrot import calculate_vpin_lite
    from .mandelbrot import calculate_cvd_refined
except Exception:
    try:
        from risk_modeling.mandelbrot import calculate_vpin_lite
        from risk_modeling.mandelbrot import calculate_cvd_refined
    except Exception:
        calculate_vpin_lite = None
        calculate_cvd_refined = None


def _normalize_ohlcv_df(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'Datetime' in df.columns:
            df = df.set_index('Datetime')
        elif 'Date' in df.columns:
            df = df.set_index('Date')
        else:
            df.index = pd.to_datetime(df.index, errors='coerce')

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("minute_df must have a DatetimeIndex or a 'Datetime'/'Date' column.")

    df = df.sort_index()
    required_columns = {'Open', 'High', 'Low', 'Close', 'Volume'}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"minute_df must contain OHLCV columns: {required_columns}")

    return df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()


def _calculate_vpin(series_close: pd.Series, series_volume: pd.Series, window: int) -> pd.Series:
    price_delta = series_close.diff().fillna(0)
    flat_volume = series_volume.where(price_delta == 0, 0.0)
    buy_volume = series_volume.where(price_delta > 0, 0.0) + 0.5 * flat_volume
    sell_volume = series_volume.where(price_delta < 0, 0.0) + 0.5 * flat_volume
    imbalance = (buy_volume - sell_volume).abs()
    volume_sum = series_volume.rolling(window=window, min_periods=1).sum().replace(0, np.nan)
    vpin = imbalance.rolling(window=window, min_periods=1).sum() / volume_sum
    return vpin.fillna(0.0)


def _calculate_cvd(series_close: pd.Series, series_volume: pd.Series, window: int) -> tuple[pd.Series, float, str]:
    price_delta = series_close.diff().fillna(0)
    direction = np.sign(price_delta).replace(0, 0.0)
    cvd = (direction * series_volume/series_volume.mean()).cumsum() if series_volume.mean() else (direction * series_volume).cumsum()
    cvd_window = min(len(cvd), window)
    if cvd_window > 1:
        x = np.arange(cvd_window)
        y = cvd.iloc[-cvd_window:].values
        slope = float(np.polyfit(x, y, 1)[0])
    else:
        slope = 0.0
    trend = 'UP' if slope > 0 else 'DOWN' if slope < 0 else 'FLAT'
    return cvd, slope, trend


def _build_volume_buckets(df: pd.DataFrame, target_bucket_volume: float):
    price_delta = df['Close'].diff().fillna(0)
    buy_volume = df['Volume'].where(price_delta > 0, 0.0) + 0.5 * df['Volume'].where(price_delta == 0, 0.0)
    sell_volume = df['Volume'].where(price_delta < 0, 0.0) + 0.5 * df['Volume'].where(price_delta == 0, 0.0)

    bucket_times = []
    bucket_imbalances = []
    bucket_vol = 0.0
    bucket_buy = 0.0
    bucket_sell = 0.0

    for ts, bv, sv, vol in zip(df.index, buy_volume, sell_volume, df['Volume']):
        if vol <= 0:
            continue

        remaining = target_bucket_volume - bucket_vol
        if vol <= remaining or remaining <= 1e-12:
            bucket_buy += bv
            bucket_sell += sv
            bucket_vol += vol
            if bucket_vol >= target_bucket_volume - 1e-12:
                bucket_imbalances.append(bucket_buy - bucket_sell)
                bucket_times.append(ts)
                bucket_buy = 0.0
                bucket_sell = 0.0
                bucket_vol = 0.0
            continue

        split_ratio = remaining / vol
        bucket_buy += bv * split_ratio
        bucket_sell += sv * split_ratio
        bucket_imbalances.append(bucket_buy - bucket_sell)
        bucket_times.append(ts)

        bucket_buy = bv * (1 - split_ratio)
        bucket_sell = sv * (1 - split_ratio)
        bucket_vol = vol * (1 - split_ratio)

    return bucket_times, bucket_imbalances


def _compute_rolling_vpin_buckets(df: pd.DataFrame, vpin_window: int = 50, window_minutes: int = 250) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=float)

    df = df.tail(window_minutes)
    total_volume = float(df['Volume'].sum())
    if total_volume <= 0:
        return pd.Series(dtype=float)

    target_bucket_volume = total_volume / max(vpin_window, 1)
    bucket_times, bucket_imbalances = _build_volume_buckets(df, target_bucket_volume)
    if not bucket_times:
        return pd.Series(dtype=float)

    vpin_values = [np.nan] * len(bucket_imbalances)
    for i in range(len(bucket_imbalances)):
        window_start = max(0, i + 1 - vpin_window)
        window_imbalances = bucket_imbalances[window_start:i + 1]
        if window_imbalances:
            vpin_values[i] = float(np.mean(np.abs(window_imbalances)) / target_bucket_volume)

    return pd.Series(vpin_values, index=bucket_times, name=f'VPIN_{vpin_window}buckets')


def compute_rolling_vpin(minute_df: pd.DataFrame, vpin_window: int = 50, window_minutes: int = 250, resample_rule: str = '5min') -> pd.Series:
    """Compute a rolling VPIN time series using equal-volume buckets.

    Parameters:
        minute_df: 1-minute OHLCV DataFrame
        vpin_window: number of volume buckets used in the rolling VPIN window
        window_minutes: minutes used to estimate target bucket volume
        resample_rule: optional frequency to resample the resulting VPIN series

    Returns:
        pd.Series indexed by bucket completion timestamps containing VPIN values.
    """
    df = _normalize_ohlcv_df(minute_df)
    vpin_series = _compute_rolling_vpin_buckets(df, vpin_window=vpin_window, window_minutes=window_minutes)

    if vpin_series.empty:
        return pd.Series(dtype=float)

    if resample_rule:
        vpin_series = vpin_series.resample(resample_rule).last().ffill()

    return vpin_series


def compute_rolling_cvd(minute_df: pd.DataFrame, resample_rule: str = '5min') -> pd.Series:
    """Compute a CVD series on resampled bars for plotting.

    Parameters:
        minute_df: 1-minute OHLCV DataFrame
        resample_rule: resampling frequency for aggregation (default '5min')

    Returns:
        pd.Series indexed by resampled bar timestamps containing cumulative delta (CVD).
    """
    df = _normalize_ohlcv_df(minute_df)
    resampled = df.resample(resample_rule, label='right', closed='right').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum',
    }).dropna(subset=['Open', 'Close'])

    if resampled.empty:
        return pd.Series(dtype=float)

    returns = resampled['Close'].pct_change().fillna(0)
    sigma = returns.std() + 1e-10
    buy_prob = norm.cdf(returns / sigma)
    delta = (2 * buy_prob - 1) * resampled['Volume']
    cvd = delta.cumsum()
    return pd.Series(cvd.values, index=resampled.index, name='CVD')


def get_hybrid_risk_signal(minute_df: pd.DataFrame, vpin_window: int = 50, cvd_window: int = 30, vpin_window_minutes: int = None) -> dict:
    """Generate a hybrid risk signal from 1-minute OHLCV.

    The function resamples 1-minute bars into 5-minute bars for Bollinger Bands
    and uses the original 1-minute bars for VPIN and CVD.

    Parameters:
        minute_df: 1-minute OHLCV DataFrame with a DatetimeIndex or 'Datetime'/'Date' column.
        vpin_window: lookback window for the VPIN calculation.
        cvd_window: lookback window for the CVD slope calculation.

    Returns:
        A structured dictionary containing the signal, band values, VPIN, CVD, and diagnostics.
    """
    df = _normalize_ohlcv_df(minute_df)
    if df.empty:
        raise ValueError("minute_df is empty after OHLCV normalization.")

    resampled = df.resample('5min', label='right', closed='right').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum',
    }).dropna(subset=['Open', 'Close'])
    if resampled.empty:
        raise ValueError('Unable to build 5-minute bars from the provided 1-minute data.')

    if _HAS_PANDAS_TA:
        bb = ta.bbands(resampled['Close'], length=20, std=2.0)
        upper_band = float(bb.filter(like='BBU').iloc[-1].iloc[0])
        middle_band = float(bb.filter(like='BBM').iloc[-1].iloc[0])
        lower_band = float(bb.filter(like='BBL').iloc[-1].iloc[0])
    else:
        middle = resampled['Close'].rolling(window=20, min_periods=1).mean()
        std = resampled['Close'].rolling(window=20, min_periods=1).std(ddof=0)
        upper_band = float((middle + 2 * std).iloc[-1])
        middle_band = float(middle.iloc[-1])
        lower_band = float((middle - 2 * std).iloc[-1])

    current_price = float(df['Close'].iloc[-1])

    vpin_series = compute_rolling_vpin(df, vpin_window=vpin_window, window_minutes=vpin_window_minutes or 250, resample_rule=None)
    if not vpin_series.empty:
        vpin_value = float(vpin_series.iloc[-1])
    else:
        # Legacy fallback if bucket-based VPIN cannot be computed
        if callable(calculate_vpin_lite):
            try:
                log_returns = np.diff(np.log(df['Close'].values))
                vpin_value = float(calculate_vpin_lite(log_returns, df['Volume'].values, window=vpin_window))
            except Exception:
                vpin_series = _calculate_vpin(df['Close'], df['Volume'], window=vpin_window)
                vpin_value = float(vpin_series.iloc[-1])
        else:
            vpin_series = _calculate_vpin(df['Close'], df['Volume'], window=vpin_window)
            vpin_value = float(vpin_series.iloc[-1])
    # Prefer mandelbrot.calculate_cvd_refined for a refined CVD slope and confidence
    if callable(calculate_cvd_refined):
        try:
            cvd_slope, cvd_confidence = calculate_cvd_refined(df, window=cvd_window)
            # Recompute a CVD series compatible with calculate_cvd_refined for display
            returns = df['Close'].pct_change().fillna(0)
            sigma = returns.std() + 1e-10
            buy_prob = norm.cdf(returns / sigma)
            delta = (2 * buy_prob - 1) * df['Volume']
            cvd_series = delta.cumsum()
            cvd_value = float(cvd_series.iloc[-1])
            cvd_trend = 'UP' if cvd_slope > 0 else 'DOWN' if cvd_slope < 0 else 'FLAT'
        except Exception as e:
            print("Error occurred while calculating refined CVD | error=%s", e)

    else:
        cvd_series, cvd_slope, cvd_trend = _calculate_cvd(df['Close'], df['Volume'], window=cvd_window)
        cvd_value = float(cvd_series.iloc[-1])

    signal = 'NEUTRAL'
    signal_reason = 'No strong entry or exit conditions met.'
    if current_price > upper_band and vpin_value > 0.70:
        signal = 'STRONG_EXIT'
        signal_reason = 'Price above 5-min upper Bollinger Band and VPIN indicates liquidity toxicity.'
    elif current_price < lower_band and vpin_value < 0.45:
        signal = 'ACCUMULATE'
        signal_reason = 'Price below 5-min lower Bollinger Band and VPIN indicates low toxicity.'

    return {
        'signal': signal,
        'signal_reason': signal_reason,
        'current_price': current_price,
        'upper_band': upper_band,
        'middle_band': middle_band,
        'lower_band': lower_band,
        'vpin': round(vpin_value, 4),
        'vpin_window': int(vpin_window),
        'cvd': round(cvd_value, 3),
        'cvd_slope': round(cvd_slope, 3),
        'cvd_trend': cvd_trend,
        'bb_source': '5-minute Bollinger Bands (20,2) from resampled 1-minute data',
        'cvd_window': int(cvd_window),
    }
