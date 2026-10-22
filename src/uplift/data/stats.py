"""Experiment statistics: average treatment effect, SRM test and covariate balance.

These are deliberately simple, dependency-light functions: they are the
sanity checks every uplift model is later compared against.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy import stats

SRM_ALPHA = 0.001  # industry convention: only flag very unlikely imbalances
SMD_THRESHOLD = 0.1  # |SMD| above this means arms are not comparable on a feature


@dataclass(frozen=True)
class EffectEstimate:
    """Difference in means between treated and control with a normal-approx CI."""

    ate: float
    se: float
    ci_low: float
    ci_high: float
    p_value: float
    treated_mean: float
    control_mean: float
    n_treated: int
    n_control: int

    @property
    def relative_lift(self) -> float:
        return self.ate / self.control_mean if self.control_mean else float("nan")

    def to_dict(self) -> dict[str, float]:
        return {**asdict(self), "relative_lift": self.relative_lift}


def average_treatment_effect(
    y: ArrayLike, treatment: ArrayLike, alpha: float = 0.05
) -> EffectEstimate:
    """ATE = E[Y|T=1] - E[Y|T=0], valid because treatment was randomised."""
    y_arr = np.asarray(y, dtype=np.float64)
    t_arr = np.asarray(treatment).astype(bool)
    if y_arr.shape != t_arr.shape:
        raise ValueError("y and treatment must have the same shape")
    y1, y0 = y_arr[t_arr], y_arr[~t_arr]
    if len(y1) < 2 or len(y0) < 2:
        raise ValueError("need at least two units in each arm")

    m1, m0 = float(y1.mean()), float(y0.mean())
    se = float(np.sqrt(y1.var(ddof=1) / len(y1) + y0.var(ddof=1) / len(y0)))
    ate = m1 - m0
    z = float(stats.norm.ppf(1 - alpha / 2))
    # With zero variance the difference is exact: p = 1 if no effect, else 0.
    p = float(2 * stats.norm.sf(abs(ate / se))) if se > 0 else float(ate == 0)
    return EffectEstimate(
        ate=ate,
        se=se,
        ci_low=ate - z * se,
        ci_high=ate + z * se,
        p_value=p,
        treated_mean=m1,
        control_mean=m0,
        n_treated=len(y1),
        n_control=len(y0),
    )


@dataclass(frozen=True)
class SRMResult:
    """Sample ratio mismatch: do observed arm sizes match the experiment design?"""

    observed: dict[str, int]
    expected_ratio: dict[str, float]
    chi2: float
    p_value: float
    alpha: float

    @property
    def passed(self) -> bool:
        return self.p_value >= self.alpha

    def to_dict(self) -> dict[str, object]:
        return {**asdict(self), "passed": self.passed}


def srm_test(
    observed: Mapping[str, int], expected_ratio: Mapping[str, float], alpha: float = SRM_ALPHA
) -> SRMResult:
    """Chi-square goodness-of-fit of arm counts against the designed allocation."""
    if set(observed) != set(expected_ratio):
        raise ValueError(f"arms differ: {sorted(observed)} vs {sorted(expected_ratio)}")
    total_ratio = sum(expected_ratio.values())
    if not np.isclose(total_ratio, 1.0):
        raise ValueError(f"expected ratios must sum to 1, got {total_ratio}")
    arms = sorted(observed)
    obs = np.array([observed[a] for a in arms], dtype=np.float64)
    exp = np.array([expected_ratio[a] for a in arms]) * obs.sum()
    chi2, p = stats.chisquare(obs, exp)
    return SRMResult(
        observed={a: int(observed[a]) for a in arms},
        expected_ratio={a: float(expected_ratio[a]) for a in arms},
        chi2=float(chi2),
        p_value=float(p),
        alpha=alpha,
    )


def standardized_mean_difference(x: ArrayLike, treatment: ArrayLike) -> float:
    """(mean_T - mean_C) / pooled SD.

    With zero within-arm variance the arms are either identical (0) or completely
    separated (+/-inf), which must be flagged as imbalance, not hidden as 0.
    """
    x_arr = np.asarray(x, dtype=np.float64)
    t_arr = np.asarray(treatment).astype(bool)
    x1, x0 = x_arr[t_arr], x_arr[~t_arr]
    diff = float(x1.mean() - x0.mean())
    pooled = float(np.sqrt((x1.var(ddof=1) + x0.var(ddof=1)) / 2))
    if pooled == 0:
        return 0.0 if diff == 0 else float(np.copysign(np.inf, diff))
    return diff / pooled


def covariate_balance(
    df: pd.DataFrame, features: Sequence[str], treatment_col: str
) -> pd.DataFrame:
    """SMD per feature (categoricals expanded one level at a time), sorted by |SMD|."""
    t = df[treatment_col].to_numpy()
    rows: list[dict[str, object]] = []
    for col in features:
        s = df[col]
        if isinstance(s.dtype, pd.CategoricalDtype) or s.dtype == object:
            for level in s.astype("category").cat.categories:
                smd = standardized_mean_difference((s == level).to_numpy(), t)
                rows.append({"feature": f"{col}={level}", "smd": smd})
        else:
            rows.append({"feature": col, "smd": standardized_mean_difference(s.to_numpy(), t)})
    out = pd.DataFrame(rows)
    out["abs_smd"] = out["smd"].abs()
    out["balanced"] = out["abs_smd"] < SMD_THRESHOLD
    return out.sort_values("abs_smd", ascending=False, ignore_index=True)
