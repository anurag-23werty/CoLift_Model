"""Experiment-integrity checks run before any modelling.

Checks (hard = pipeline fails, soft = reported only):
  * schema     (hard) Pandera contract: types, allowed values, cross-column rules
  * row_count  (hard) matches the published dataset size
  * srm        (hard) arm sizes match the randomisation design
  * leakage    (hard) no post-treatment column in the feature list
  * balance    (soft) |SMD| < 0.1 for every feature, i.e. arms are comparable
  * ate        (info) difference-in-means effect per outcome and per arm

Usage:
    python -m uplift.data.validate --dataset hillstrom
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pandera.errors

from uplift.config import load_config
from uplift.data.datasets import (
    SPECS,
    TREATMENT,
    ExperimentSpec,
    LeakageError,
    arm_labels,
    assert_no_leakage,
    load_raw,
)
from uplift.data.schemas import validate_schema
from uplift.data.stats import average_treatment_effect, covariate_balance, srm_test
from uplift.logging_utils import get_logger

log = get_logger(__name__)


class DataValidationError(RuntimeError):
    """One or more hard validation checks failed."""


def _ate_report(df: pd.DataFrame, spec: ExperimentSpec) -> dict[str, Any]:
    arms = arm_labels(df, spec)
    report: dict[str, Any] = {}
    for outcome in spec.outcomes:
        per_outcome: dict[str, Any] = {
            "any_treatment": average_treatment_effect(df[outcome], df[TREATMENT]).to_dict()
        }
        treated_arms = [a for a in spec.arm_ratios if a != spec.control_arm]
        if len(treated_arms) > 1:
            for arm in treated_arms:
                mask = arms.isin([arm, spec.control_arm])
                per_outcome[arm] = average_treatment_effect(
                    df.loc[mask, outcome], arms[mask] == arm
                ).to_dict()
        report[outcome] = per_outcome
    return report


def run_checks(df: pd.DataFrame, spec: ExperimentSpec) -> dict[str, Any]:
    """Run every check and return a JSON-serialisable report (never raises)."""
    checks: dict[str, Any] = {}

    try:
        validate_schema(df, spec.name)
        checks["schema"] = {"passed": True}
    except pandera.errors.SchemaErrors as exc:
        cases = exc.failure_cases.head(20).astype(str).to_dict(orient="records")
        checks["schema"] = {"passed": False, "failure_cases": cases}

    checks["row_count"] = {
        "passed": len(df) == spec.expected_rows,
        "observed": len(df),
        "expected": spec.expected_rows,
    }

    counts = arm_labels(df, spec).value_counts().to_dict()
    checks["srm"] = srm_test({str(k): int(v) for k, v in counts.items()}, spec.arm_ratios).to_dict()

    try:
        assert_no_leakage(spec.features, spec)
        checks["leakage"] = {"passed": True, "features": list(spec.features)}
    except LeakageError as exc:
        checks["leakage"] = {"passed": False, "error": str(exc)}

    balance = covariate_balance(df, spec.features, TREATMENT)
    checks["balance"] = {
        "passed": bool(balance["balanced"].all()),
        "max_abs_smd": float(balance["abs_smd"].max()),
        "per_feature": balance[["feature", "smd"]].round(5).to_dict(orient="records"),
    }

    hard = ("schema", "row_count", "srm", "leakage")
    return {
        "dataset": spec.name,
        "n_rows": len(df),
        "treatment_rate": float(df[TREATMENT].mean()),
        "outcome_rates": {o: float(df[o].mean()) for o in spec.outcomes},
        "passed": all(checks[k]["passed"] for k in hard),
        "checks": checks,
        "ate": _ate_report(df, spec),
    }


def validate(name: str, raw_dir: Path, reports_dir: Path) -> dict[str, Any]:
    spec = SPECS[name]
    df = load_raw(name, raw_dir)
    report = run_checks(df, spec)

    out = reports_dir / "validation" / f"{name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    log.info("Wrote %s", out)

    for check, result in report["checks"].items():
        log.info("  %-10s %s", check, "PASS" if result["passed"] else "FAIL")
    if not report["passed"]:
        failed = [k for k, v in report["checks"].items() if not v["passed"] and k != "balance"]
        raise DataValidationError(f"{name}: hard checks failed: {failed} (see {out})")
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=[*SPECS, "all"], default="all")
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)
    cfg = load_config(args.config) if args.config else load_config()
    names = list(SPECS) if args.dataset == "all" else [args.dataset]
    for name in names:
        validate(name, cfg.paths.raw_dir, cfg.paths.reports_dir)


if __name__ == "__main__":
    main()
