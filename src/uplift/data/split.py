"""Reproducible train / val / test split, stratified on treatment x outcome.

Stratifying on the treatment-outcome cell (not just the outcome) keeps the
treated/control ratio *and* the conversion rate inside each arm identical
across splits. With ~0.3% conversion on Criteo, an unstratified split can
leave the control arm of the test set with too few positives for a stable
Qini curve.

Criteo dev runs also get a stratified sample (`criteo_sample_frac`), so
experiments run on a laptop while the full data is kept for the final benchmark.

Usage:
    python -m uplift.data.split --dataset hillstrom
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from uplift.config import Config, load_config
from uplift.data.datasets import SPECS, TREATMENT, ExperimentSpec, load_raw
from uplift.logging_utils import get_logger

log = get_logger(__name__)

ROW_ID = "row_id"
SPLITS = ("train", "val", "test")


def strata(df: pd.DataFrame, spec: ExperimentSpec) -> pd.Series:
    """Stratum label = arm x (primary binary outcome)."""
    primary = "conversion" if "conversion" in spec.outcomes else spec.outcomes[0]
    arm = cast("pd.Series[str]", df[spec.arm_col].astype(str))
    outcome = cast("pd.Series[str]", df[primary].astype(str))
    return arm + "|" + outcome


def stratified_sample(
    df: pd.DataFrame, spec: ExperimentSpec, frac: float, seed: int
) -> pd.DataFrame:
    if frac >= 1.0:
        return df
    sample, _ = train_test_split(df, train_size=frac, stratify=strata(df, spec), random_state=seed)
    return pd.DataFrame(sample).sort_values(ROW_ID, ignore_index=True)


def split_frame(
    df: pd.DataFrame, spec: ExperimentSpec, val_size: float, test_size: float, seed: int
) -> dict[str, pd.DataFrame]:
    """Split into train/val/test with fractions of the full frame."""
    rest, test = train_test_split(
        df, test_size=test_size, stratify=strata(df, spec), random_state=seed
    )
    train, val = train_test_split(
        rest,
        test_size=val_size / (1 - test_size),
        stratify=strata(rest, spec),
        random_state=seed,
    )
    return {
        name: part.sort_values(ROW_ID, ignore_index=True)
        for name, part in zip(SPLITS, (train, val, test), strict=True)
    }


def frame_fingerprint(df: pd.DataFrame) -> str:
    """Content hash that is stable across runs and machines (used to prove reproducibility)."""
    hashed = pd.util.hash_pandas_object(df, index=False).to_numpy()
    return hashlib.sha256(hashed.tobytes()).hexdigest()


def _summary(parts: dict[str, pd.DataFrame], spec: ExperimentSpec) -> dict[str, Any]:
    total = sum(len(p) for p in parts.values())
    out: dict[str, Any] = {}
    for name, part in parts.items():
        out[name] = {
            "rows": len(part),
            "fraction": round(len(part) / total, 4),
            "treatment_rate": float(part[TREATMENT].mean()),
            **{f"{o}_rate": float(part[o].mean()) for o in spec.outcomes},
            "sha256": frame_fingerprint(part),
        }
    return out


def make_splits(name: str, cfg: Config) -> dict[str, Any]:
    spec = SPECS[name]
    df = load_raw(name, cfg.paths.raw_dir)
    df.insert(0, ROW_ID, np.arange(len(df), dtype=np.int64))

    frac = cfg.data.criteo_sample_frac if name == "criteo" else 1.0
    df = stratified_sample(df, spec, frac, cfg.seed)
    parts = split_frame(df, spec, cfg.data.val_size, cfg.data.test_size, cfg.seed)

    out_dir = cfg.paths.processed_dir / name
    out_dir.mkdir(parents=True, exist_ok=True)
    for split_name, part in parts.items():
        part.to_parquet(out_dir / f"{split_name}.parquet", index=False)

    summary = {"dataset": name, "seed": cfg.seed, "sample_frac": frac, **_summary(parts, spec)}
    report = cfg.paths.reports_dir / "splits" / f"{name}.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(summary, indent=2))
    for split_name in SPLITS:
        s = summary[split_name]
        log.info(
            "%s/%-5s rows=%-9d treat=%.4f sha=%s",
            name, split_name, s["rows"], s["treatment_rate"], s["sha256"][:12],
        )  # fmt: skip
    return summary


def load_split(name: str, split: str, processed_dir: Path) -> pd.DataFrame:
    path = processed_dir / name / f"{split}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `make split` first")
    return pd.read_parquet(path)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=[*SPECS, "all"], default="all")
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)
    cfg = load_config(args.config) if args.config else load_config()
    for name in list(SPECS) if args.dataset == "all" else [args.dataset]:
        make_splits(name, cfg)


if __name__ == "__main__":
    main()
