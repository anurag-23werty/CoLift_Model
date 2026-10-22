"""Reusable EDA computations (the notebook only plots what these return)."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from uplift.data.datasets import ExperimentSpec
from uplift.data.stats import average_treatment_effect


def outcome_rates_by_arm(df: pd.DataFrame, spec: ExperimentSpec) -> pd.DataFrame:
    """Mean of every outcome per experiment arm, plus arm size."""
    grouped = df.groupby(df[spec.arm_col].astype(str), observed=True)
    rates = grouped[list(spec.outcomes)].mean()
    rates["n"] = grouped.size()
    return rates


def effect_by_segment(
    df: pd.DataFrame,
    spec: ExperimentSpec,
    by: Sequence[str],
    outcome: str,
    arms: Sequence[str] | None = None,
) -> pd.DataFrame:
    """ATE of each treated arm vs control, within every level of every `by` column.

    Large differences between levels (heterogeneous effects) are the reason a
    per-user uplift model can beat sending the treatment to everyone.
    """
    labels = df[spec.arm_col].astype(str)
    if arms is None:
        arms = [a for a in spec.arm_ratios if a != spec.control_arm]
    rows: list[dict[str, object]] = []
    for col in by:
        for level, idx in df.groupby(col, observed=True).groups.items():
            seg = df.loc[idx]
            seg_labels = labels.loc[idx]
            for arm in arms:
                mask = seg_labels.isin([arm, spec.control_arm])
                est = average_treatment_effect(seg.loc[mask, outcome], seg_labels[mask] == arm)
                rows.append(
                    {
                        "feature": col,
                        "level": str(level),
                        "arm": arm,
                        "ate": est.ate,
                        "ci_low": est.ci_low,
                        "ci_high": est.ci_high,
                        "n": int(mask.sum()),
                    }
                )
    return pd.DataFrame(rows)
