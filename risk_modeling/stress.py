"""Market stress scoring model.

This module measures realized market stress rather than latent structural
fragility. It converts aligned market indicators into rolling percentile
scores, combines them using configurable weights, and returns a bounded
0-to-100 stress score with detailed diagnostics.

Higher values must consistently represent greater market stress.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class StressConfig:
    """Configuration for the market stress model."""

    realized_vol_weight: float = 0.20
    drawdown_weight: float = 0.20
    vix_weight: float = 0.20
    hy_spread_weight: float = 0.20
    ig_spread_weight: float = 0.10
    credit_risk_premium_weight: float = 0.10

    lookback: int = 504
    min_observations: int = 126

    # Optional confirmation multiplier when several stress signals are extreme.
    confirmation_percentile: float = 80.0
    confirmation_count: int = 3
    confirmation_multiplier: float = 1.10

    # Logistic transformation from the weighted percentile score to 0-100.
    logistic_midpoint: float = 50.0
    logistic_scale: float = 10.0

    watch_threshold: float = 60.0
    elevated_threshold: float = 75.0
    critical_threshold: float = 85.0

    def __post_init__(self) -> None:
        weights = np.asarray(
            [
                self.realized_vol_weight,
                self.drawdown_weight,
                self.vix_weight,
                self.hy_spread_weight,
                self.ig_spread_weight,
                self.credit_risk_premium_weight,
            ],
            dtype=float,
        )

        if not np.all(np.isfinite(weights)):
            raise ValueError("All component weights must be finite.")
        if np.any(weights < 0.0):
            raise ValueError("Component weights cannot be negative.")
        if not np.isclose(weights.sum(), 1.0, atol=1e-8):
            raise ValueError(
                f"Component weights must sum to 1.0; received {weights.sum():.6f}."
            )
        if self.lookback < 2:
            raise ValueError("lookback must be at least 2.")
        if not 2 <= self.min_observations <= self.lookback:
            raise ValueError(
                "min_observations must be at least 2 and cannot exceed lookback."
            )
        if self.logistic_scale <= 0.0:
            raise ValueError("logistic_scale must be greater than zero.")
        if not 0.0 <= self.confirmation_percentile <= 100.0:
            raise ValueError("confirmation_percentile must be between 0 and 100.")
        if not 1 <= self.confirmation_count <= 5:
            raise ValueError("confirmation_count must be between 1 and 5.")
        if self.confirmation_multiplier < 1.0:
            raise ValueError("confirmation_multiplier must be at least 1.0.")

        thresholds = (
            self.watch_threshold,
            self.elevated_threshold,
            self.critical_threshold,
        )
        if not all(0.0 <= value <= 100.0 for value in thresholds):
            raise ValueError("Alert thresholds must be between 0 and 100.")
        if not (
            self.watch_threshold
            < self.elevated_threshold
            < self.critical_threshold
        ):
            raise ValueError("Thresholds must satisfy watch < elevated < critical.")


def _to_1d_float_array(
    values: Sequence[float] | np.ndarray,
    name: str,
) -> np.ndarray:
    """Convert input to a non-empty one-dimensional float array."""

    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if array.size == 0:
        raise ValueError(f"{name} cannot be empty.")
    return array


def _latest_finite(array: np.ndarray, name: str) -> float:
    """Return the most recent finite observation."""

    positions = np.flatnonzero(np.isfinite(array))
    if positions.size == 0:
        raise ValueError(f"{name} does not contain any finite values.")
    return float(array[positions[-1]])


def _percentile_rank(
    current_value: float,
    history: np.ndarray,
    *,
    lookback: int,
    min_observations: int,
) -> float:
    """Return the empirical midpoint percentile rank from 0 to 100."""

    if not np.isfinite(current_value):
        raise ValueError("The current value must be finite.")

    sample = np.asarray(history, dtype=float).reshape(-1)[-lookback:]
    sample = sample[np.isfinite(sample)]
    if sample.size < min_observations:
        raise ValueError(
            "Insufficient observations for percentile ranking: "
            f"received {sample.size}, required {min_observations}."
        )

    below = np.count_nonzero(sample < current_value)
    equal = np.count_nonzero(sample == current_value)
    rank = 100.0 * (below + 0.5 * equal) / sample.size
    return float(np.clip(rank, 0.0, 100.0))


def _logistic_score(raw_score: float, midpoint: float, scale: float) -> float:
    """Map a raw percentile score to a bounded 0-to-100 score."""

    exponent = np.clip(-(raw_score - midpoint) / scale, -700.0, 700.0)
    return float(100.0 / (1.0 + np.exp(exponent)))


def _risk_level(score: float, config: StressConfig) -> str:
    """Classify the final stress score."""

    if score >= config.critical_threshold:
        return "critical"
    if score >= config.elevated_threshold:
        return "elevated"
    if score >= config.watch_threshold:
        return "watch"
    return "normal"


def calculate_stress_score(
    *,
    realized_vol_history: Sequence[float] | np.ndarray,
    drawdown_history: Sequence[float] | np.ndarray,
        vix_history: Sequence[float] | np.ndarray,
        hy_spread_history: Sequence[float] | np.ndarray,
        ig_spread_history: Sequence[float] | np.ndarray,
        credit_risk_premium_history: Sequence[float] | np.ndarray,
    config: Optional[StressConfig] = None,
) -> dict[str, Any]:
    """Calculate a normalized 0-to-100 realized market stress score.

    Parameters
    ----------
    realized_vol_history:
        Historical realized-volatility series. Higher values indicate stress.
        Use a consistent annualized or non-annualized convention.
    drawdown_history:
        Historical drawdowns, normally zero or negative. More-negative values
        indicate deeper losses and are internally converted to positive stress.
    vix_history:
        Historical VIX observations. Higher values indicate stress.
    hy_spread_history:
        Historical high-yield option-adjusted spread. Higher values indicate
        tighter financial conditions and greater stress.
    ig_spread_history:
        Historical investment-grade option-adjusted spread.
    credit_risk_premium_history:
        Historical high-yield minus investment-grade spread. Higher values
        indicate wider compensation for high-yield credit risk.
    config:
        Optional StressConfig instance.

    Returns
    -------
    dict[str, Any]
        Final score, raw scores, risk classification, alert flags, component
        percentiles, weighted contributions, current signals, and diagnostics.

    Notes
    -----
    Inputs should be aligned to the same dates and frequency. For a valid
    backtest, compute each date using only historical information available at
    that date to avoid look-ahead bias.
    """

    cfg = config or StressConfig()

    realized_vol = _to_1d_float_array(
        realized_vol_history, "realized_vol_history"
    )
    drawdown = _to_1d_float_array(drawdown_history, "drawdown_history")
    vix = _to_1d_float_array(vix_history, "vix_history")
    hy_spread = _to_1d_float_array(hy_spread_history, "hy_spread_history")
    ig_spread = _to_1d_float_array(ig_spread_history, "ig_spread_history")
    credit_risk_premium = _to_1d_float_array(
        credit_risk_premium_history, "credit_risk_premium_history"
    )

    latest_inputs = {
        "realized_vol": _latest_finite(realized_vol, "realized_vol_history"),
        "drawdown": _latest_finite(drawdown, "drawdown_history"),
        "vix": _latest_finite(vix, "vix_history"),
        "hy_spread": _latest_finite(hy_spread, "hy_spread_history"),
        "ig_spread": _latest_finite(ig_spread, "ig_spread_history"),
        "credit_risk_premium": _latest_finite(
            credit_risk_premium, "credit_risk_premium_history"
        ),
    }

    transformed_histories = {
        "realized_vol": realized_vol,
        "drawdown": np.maximum(-drawdown, 0.0),
        "vix": vix,
        "hy_spread": hy_spread,
        "ig_spread": ig_spread,
        "credit_risk_premium": credit_risk_premium,
    }
    current_signals = {
        "realized_vol": latest_inputs["realized_vol"],
        "drawdown_depth": max(-latest_inputs["drawdown"], 0.0),
        "vix": latest_inputs["vix"],
        "hy_spread": latest_inputs["hy_spread"],
        "ig_spread": latest_inputs["ig_spread"],
        "credit_risk_premium": latest_inputs["credit_risk_premium"],
    }

    percentile_inputs = {
        "realized_vol": current_signals["realized_vol"],
        "drawdown": current_signals["drawdown_depth"],
        "vix": current_signals["vix"],
        "hy_spread": current_signals["hy_spread"],
        "ig_spread": current_signals["ig_spread"],
        "credit_risk_premium": current_signals["credit_risk_premium"],
    }

    percentiles = {
        name: _percentile_rank(
            percentile_inputs[name],
            transformed_histories[name],
            lookback=cfg.lookback,
            min_observations=cfg.min_observations,
        )
        for name in percentile_inputs
    }

    weights = {
        "realized_vol": cfg.realized_vol_weight,
        "drawdown": cfg.drawdown_weight,
        "vix": cfg.vix_weight,
        "hy_spread": cfg.hy_spread_weight,
        "ig_spread": cfg.ig_spread_weight,
        "credit_risk_premium": cfg.credit_risk_premium_weight,
    }
    weighted_components = {
        name: weights[name] * percentiles[name] for name in percentiles
    }
    base_raw_score = float(sum(weighted_components.values()))
    extreme_components = [
        name
        for name, percentile in percentiles.items()
        if percentile >= cfg.confirmation_percentile
    ]
    confirmation_applied = len(extreme_components) >= cfg.confirmation_count
    adjusted_raw_score = base_raw_score
    if confirmation_applied:
        adjusted_raw_score *= cfg.confirmation_multiplier
    adjusted_raw_score = float(np.clip(adjusted_raw_score, 0.0, 100.0))

    score = _logistic_score(
        adjusted_raw_score,
        cfg.logistic_midpoint,
        cfg.logistic_scale,
    )
    risk_level = _risk_level(score, cfg)

    return {
        "score": score,
        "raw_score": adjusted_raw_score,
        "base_raw_score": base_raw_score,
        "risk_level": risk_level,
        "is_watch": risk_level == "watch",
        "is_elevated": risk_level == "elevated",
        "is_critical": risk_level == "critical",
        "confirmation_applied": confirmation_applied,
        "confirmation_multiplier_applied": (
            cfg.confirmation_multiplier if confirmation_applied else 1.0
        ),
        "extreme_components": extreme_components,
        "component_percentiles": percentiles,
        "weighted_components": weighted_components,
        "current_signals": current_signals,
        "latest_inputs": latest_inputs,
        "thresholds": {
            "watch": cfg.watch_threshold,
            "elevated": cfg.elevated_threshold,
            "critical": cfg.critical_threshold,
        },
        "observations": {
            "normalization_lookback": cfg.lookback,
            "minimum_required": cfg.min_observations,
        },
    }


__all__ = ["StressConfig", "calculate_stress_score"]
