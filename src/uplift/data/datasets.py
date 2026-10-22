"""Experiment specs: what each dataset's design looks like and which columns are which.

Keeping this in one place is what lets the leakage guard work: a column is
either a pre-treatment feature or it is not, and that is declared here, not
decided ad hoc in a notebook.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

TREATMENT = "treatment"


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    arm_col: str
    control_arm: str
    arm_ratios: dict[str, float]
    features: tuple[str, ...]
    categorical_features: tuple[str, ...]
    outcomes: tuple[str, ...]
    post_treatment: tuple[str, ...]
    expected_rows: int

    @property
    def numeric_features(self) -> tuple[str, ...]:
        return tuple(f for f in self.features if f not in self.categorical_features)


HILLSTROM_SPEC = ExperimentSpec(
    name="hillstrom",
    arm_col="segment",
    control_arm="No E-Mail",
    arm_ratios={"Mens E-Mail": 1 / 3, "Womens E-Mail": 1 / 3, "No E-Mail": 1 / 3},
    # history_segment is just a binned copy of history, so it is left out of the features.
    features=("recency", "history", "mens", "womens", "zip_code", "newbie", "channel"),
    categorical_features=("zip_code", "channel"),
    outcomes=("visit", "conversion", "spend"),
    post_treatment=("visit", "conversion", "spend"),
    expected_rows=64_000,
)

CRITEO_SPEC = ExperimentSpec(
    name="criteo",
    arm_col=TREATMENT,
    control_arm="0",
    arm_ratios={"1": 0.85, "0": 0.15},
    features=tuple(f"f{i}" for i in range(12)),
    categorical_features=(),
    outcomes=("visit", "conversion"),
    # `exposure` is recorded after assignment: the classic leakage trap on this dataset.
    post_treatment=("visit", "conversion", "exposure"),
    expected_rows=13_979_592,
)

SPECS: dict[str, ExperimentSpec] = {s.name: s for s in (HILLSTROM_SPEC, CRITEO_SPEC)}


class LeakageError(ValueError):
    """Raised when a post-treatment column is about to be used as a feature."""


def assert_no_leakage(features: tuple[str, ...] | list[str], spec: ExperimentSpec) -> None:
    leaked = sorted(set(features) & ({*spec.post_treatment, TREATMENT, spec.arm_col}))
    if leaked:
        raise LeakageError(f"post-treatment or assignment columns used as features: {leaked}")


def add_treatment(df: pd.DataFrame, spec: ExperimentSpec) -> pd.DataFrame:
    """Add a binary `treatment` column (any treatment arm vs control)."""
    if spec.arm_col == TREATMENT:
        return df
    out = df.copy()
    out[TREATMENT] = (out[spec.arm_col].astype(str) != spec.control_arm).astype("int8")
    return out


def arm_labels(df: pd.DataFrame, spec: ExperimentSpec) -> pd.Series:
    """Arm of each row as a string, matching the keys of `spec.arm_ratios`."""
    return df[spec.arm_col].astype(str)


def load_raw(name: str, raw_dir: Path) -> pd.DataFrame:
    """Read the raw parquet written by `uplift.data.download` and add `treatment`."""
    spec = SPECS[name]
    path = raw_dir / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `make data-download` first")
    return add_treatment(pd.read_parquet(path), spec)
