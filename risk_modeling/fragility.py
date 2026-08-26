from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class FragilityConfig:
    """Configuration for the market fragility model."""

    # Component weights must sum to 1.0.
    flow_weight: float = 0.35
    breadth_weight: float = 0.25
    hurst_weight: float = 0.15
    concentration_weight: float = 0.15
    drawdown_accel_weight: float = 0.10

    # Historical normalization settings.
    lookback: int = 504
    min_observations: int = 126

    # Hurst value below which trend persistence is considered weak.
    hurst_reference: float = 0.55

    # Divergence condition: positive price return with weakening internals.
    divergence_multiplier: float = 1.15
    positive_return_threshold: float = 0.0

    # Logistic transformation of the weighted percentile score.
    logistic_midpoint: float = 50.0
    logistic_scale: float = 10.0

    # Alert thresholds on the final 0-to-100 score.
    watch_threshold: float = 60.0
    elevated_threshold: float = 75.0
    critical_threshold: float = 85.0

    def __post_init__(self) -> None:
        weights = np.array(
            [
                self.flow_weight,
                self.breadth_weight,
                self.hurst_weight,
                self.concentration_weight,
                self.drawdown_accel_weight,
            ],
            dtype=float,
        )

        if not np.all(np.isfinite(weights)):
            raise ValueError("All component weights must be finite.")

        if np.any(weights < 0):
            raise ValueError("Component weights cannot be negative.")

        if not np.isclose(weights.sum(), 1.0, atol=1e-8):
            raise ValueError(
                f"Component weights must sum to 1.0; received {weights.sum():.6f}."
            )

        if self.lookback < 2:
            raise ValueError("lookback must be at least 2.")

        if self.min_observations < 2:
            raise ValueError("min_observations must be at least 2.")

        if self.min_observations > self.lookback:
            raise ValueError("min_observations cannot exceed lookback.")

        if self.logistic_scale <= 0:
            raise ValueError("logistic_scale must be greater than zero.")

        if self.divergence_multiplier < 1.0:
            raise ValueError("divergence_multiplier must be at least 1.0.")

        thresholds = (
            self.watch_threshold,
            self.elevated_threshold,
            self.critical_threshold,
        )

        if not all(0.0 <= level <= 100.0 for level in thresholds):
            raise ValueError("Alert thresholds must be between 0 and 100.")

        if not (
            self.watch_threshold
            < self.elevated_threshold
            < self.critical_threshold
        ):
            raise ValueError(
                "Thresholds must satisfy watch < elevated < critical."
            )


def _to_finite_array(
    values: Sequence[float] | np.ndarray,
    name: str,
) -> np.ndarray:
    """Convert an input sequence to a one-dimensional float array."""

    array = np.asarray(values, dtype=float)

    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")

    if array.size == 0:
        raise ValueError(f"{name} cannot be empty.")

    return array


def _latest_finite_value(array: np.ndarray, name: str) -> float:
    """Return the latest finite value from an array."""

    finite_positions = np.flatnonzero(np.isfinite(array))

    if finite_positions.size == 0:
        raise ValueError(f"{name} does not contain any finite values.")

    return float(array[finite_positions[-1]])


def _percentile_rank(
    current_value: float,
    history: Sequence[float] | np.ndarray,
    *,
    lookback: int,
    min_observations: int,
) -> float:
    """
    Calculate an empirical percentile rank from 0 to 100.

    Ties receive their midpoint percentile, which avoids unstable jumps
    when a signal contains many repeated values.
    """

    if not np.isfinite(current_value):
        raise ValueError("The current value must be finite.")

    historical_array = np.asarray(history, dtype=float).reshape(-1)
    historical_array = historical_array[-lookback:]
    historical_array = historical_array[np.isfinite(historical_array)]

    if historical_array.size < min_observations:
        raise ValueError(
            "Insufficient historical observations for percentile ranking: "
            f"received {historical_array.size}, "
            f"required at least {min_observations}."
        )

    below = np.count_nonzero(historical_array < current_value)
    equal = np.count_nonzero(historical_array == current_value)

    percentile = 100.0 * (
        below + 0.5 * equal
    ) / historical_array.size

    return float(np.clip(percentile, 0.0, 100.0))


def _logistic_score(
    raw_score: float,
    midpoint: float,
    scale: float,
) -> float:
    """Map a raw percentile score to a bounded 0-to-100 score."""

    exponent = np.clip(
        -(raw_score - midpoint) / scale,
        -700.0,
        700.0,
    )

    return float(100.0 / (1.0 + np.exp(exponent)))


def _risk_level(
    score: float,
    config: FragilityConfig,
) -> str:
    """Classify the final fragility score."""

    if score >= config.critical_threshold:
        return "critical"

    if score >= config.elevated_threshold:
        return "elevated"

    if score >= config.watch_threshold:
        return "watch"

    return "normal"


def interpret_fragility_score(score: float) -> tuple[str, str]:
    """Return the display interpretation and icon for a 0-to-100 score."""
    bounded_score = float(np.clip(score, 0.0, 100.0))
    if bounded_score < 30.0:
        return "Healthy", "✅"
    if bounded_score < 50.0:
        return "Normal", "🟢"
    if bounded_score < 70.0:
        return "Watch", "⚠️"
    if bounded_score < 85.0:
        return "Elevated", "🟠"
    return "Critical", "🚨"


def calculate_fragility_score(
    *,
    returns_1m: Sequence[float] | np.ndarray,
    cmf_history: Sequence[float] | np.ndarray,
    breadth_slope_history: Sequence[float] | np.ndarray,
    hurst_history: Sequence[float] | np.ndarray,
    concentration_history: Sequence[float] | np.ndarray,
    drawdown_accel_history: Sequence[float] | np.ndarray,
    config: Optional[FragilityConfig] = None,
) -> dict[str, Any]:
    """
    Calculate a normalized, regime-aware market fragility score.

    Parameters
    ----------
    returns_1m:
        Historical one-month returns. The latest observation is used
        to detect a narrow-rally divergence.

    cmf_history:
        Historical Chaikin Money Flow observations. Negative CMF means
        selling pressure. The score uses max(-CMF, 0).

    breadth_slope_history:
        Historical market-breadth slopes. Negative slopes indicate
        deteriorating participation. The score uses -breadth_slope.

    hurst_history:
        Historical Hurst exponent observations. Values below the configured
        reference level indicate weak trend persistence. The score uses
        max(hurst_reference - H, 0).

    concentration_history:
        Historical concentration measurements, such as top-10 index weight
        or a normalized Herfindahl-Hirschman Index. Higher values must mean
        greater concentration risk.

    drawdown_accel_history:
        Historical drawdown-acceleration measurements. The convention must
        be that more negative values indicate faster deterioration. The
        score therefore uses -drawdown_acceleration.

    config:
        Optional model configuration.

    Returns
    -------
    dict
        Final score, raw score, risk level, alert flags, component
        percentiles, current signal values, and diagnostic information.

    Notes
    -----
    All histories should use the same frequency and should be aligned to
    the same market dates before this function is called.
    """

    cfg = config or FragilityConfig()

    returns = _to_finite_array(returns_1m, "returns_1m")
    cmf = _to_finite_array(cmf_history, "cmf_history")
    breadth = _to_finite_array(
        breadth_slope_history,
        "breadth_slope_history",
    )
    hurst = _to_finite_array(hurst_history, "hurst_history")
    concentration = _to_finite_array(
        concentration_history,
        "concentration_history",
    )
    drawdown_accel = _to_finite_array(
        drawdown_accel_history,
        "drawdown_accel_history",
    )

    latest_return = _latest_finite_value(returns, "returns_1m")
    latest_cmf = _latest_finite_value(cmf, "cmf_history")
    latest_breadth = _latest_finite_value(
        breadth,
        "breadth_slope_history",
    )
    latest_hurst = _latest_finite_value(hurst, "hurst_history")
    latest_concentration = _latest_finite_value(
        concentration,
        "concentration_history",
    )
    latest_drawdown_accel = _latest_finite_value(
        drawdown_accel,
        "drawdown_accel_history",
    )

    # Convert each metric so that a larger value always means more fragility.
    downside_flow_history = np.maximum(-cmf, 0.0)
    breadth_weakness_history = np.maximum(-breadth, 0.0)
    weak_memory_history = np.maximum(
        cfg.hurst_reference - hurst,
        0.0,
    )
    drawdown_deterioration_history = np.maximum(
        -drawdown_accel,
        0.0,
    )

    current_signals = {
        "downside_flow": max(-latest_cmf, 0.0),
        "breadth_weakness": max(-latest_breadth, 0.0),
        "weak_memory": max(
            cfg.hurst_reference - latest_hurst,
            0.0,
        ),
        "concentration": latest_concentration,
        "drawdown_deterioration": max(
            -latest_drawdown_accel,
            0.0,
        ),
    }

    percentiles = {
        "flow": _percentile_rank(
            current_signals["downside_flow"],
            downside_flow_history,
            lookback=cfg.lookback,
            min_observations=cfg.min_observations,
        ),
        "breadth": _percentile_rank(
            current_signals["breadth_weakness"],
            breadth_weakness_history,
            lookback=cfg.lookback,
            min_observations=cfg.min_observations,
        ),
        "hurst": _percentile_rank(
            current_signals["weak_memory"],
            weak_memory_history,
            lookback=cfg.lookback,
            min_observations=cfg.min_observations,
        ),
        "concentration": _percentile_rank(
            current_signals["concentration"],
            concentration,
            lookback=cfg.lookback,
            min_observations=cfg.min_observations,
        ),
        "drawdown_acceleration": _percentile_rank(
            current_signals["drawdown_deterioration"],
            drawdown_deterioration_history,
            lookback=cfg.lookback,
            min_observations=cfg.min_observations,
        ),
    }

    weighted_components = {
        "flow": cfg.flow_weight * percentiles["flow"],
        "breadth": cfg.breadth_weight * percentiles["breadth"],
        "hurst": cfg.hurst_weight * percentiles["hurst"],
        "concentration": (
            cfg.concentration_weight
            * percentiles["concentration"]
        ),
        "drawdown_acceleration": (
            cfg.drawdown_accel_weight
            * percentiles["drawdown_acceleration"]
        ),
    }

    base_raw_score = float(sum(weighted_components.values()))

    # Classic narrow-rally divergence:
    # price remains positive while flows and breadth deteriorate.
    is_narrow_rally_divergence = bool(
        latest_return > cfg.positive_return_threshold
        and latest_cmf < 0.0
        and latest_breadth < 0.0
    )

    adjusted_raw_score = base_raw_score

    if is_narrow_rally_divergence:
        adjusted_raw_score *= cfg.divergence_multiplier

    # Keep the adjusted percentile score interpretable and bounded.
    adjusted_raw_score = float(
        np.clip(adjusted_raw_score, 0.0, 100.0)
    )

    final_score = _logistic_score(
        raw_score=adjusted_raw_score,
        midpoint=cfg.logistic_midpoint,
        scale=cfg.logistic_scale,
    )

    risk_level = _risk_level(final_score, cfg)
    interpretation, icon = interpret_fragility_score(final_score)

    return {
        "score": final_score,
        "interpretation": interpretation,
        "icon": icon,
        "raw_score": adjusted_raw_score,
        "base_raw_score": base_raw_score,
        "risk_level": risk_level,
        "is_watch": risk_level == "watch",
        "is_elevated": risk_level == "elevated",
        "is_critical": risk_level == "critical",
        "is_narrow_rally_divergence": is_narrow_rally_divergence,
        "divergence_multiplier_applied": (
            cfg.divergence_multiplier
            if is_narrow_rally_divergence
            else 1.0
        ),
        "component_percentiles": percentiles,
        "weighted_components": weighted_components,
        "current_signals": current_signals,
        "latest_inputs": {
            "return_1m": latest_return,
            "cmf": latest_cmf,
            "breadth_slope": latest_breadth,
            "hurst": latest_hurst,
            "concentration": latest_concentration,
            "drawdown_acceleration": latest_drawdown_accel,
        },
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