import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from uplift.data.datasets import CRITEO_SPEC, HILLSTROM_SPEC, LeakageError, assert_no_leakage
from uplift.data.validate import DataValidationError, run_checks, validate


def _spec_for(df: pd.DataFrame):  # type: ignore[no-untyped-def]
    return replace(HILLSTROM_SPEC, expected_rows=len(df))


def test_clean_data_passes(hillstrom_df: pd.DataFrame) -> None:
    report = run_checks(hillstrom_df, _spec_for(hillstrom_df))
    assert report["passed"], report["checks"]
    assert report["checks"]["balance"]["passed"]
    visit_ate = report["ate"]["visit"]["any_treatment"]
    assert visit_ate["ci_low"] > 0  # synthetic data has a real email effect
    assert set(report["ate"]["visit"]) == {"any_treatment", "Mens E-Mail", "Womens E-Mail"}
    json.dumps(report)  # must be serialisable for the report file


def test_schema_violation_detected(hillstrom_df: pd.DataFrame) -> None:
    broken = hillstrom_df.copy()
    broken.loc[broken.index[0], ["visit", "conversion", "spend"]] = [0, 1, 50.0]
    report = run_checks(broken, _spec_for(broken))
    assert not report["checks"]["schema"]["passed"]
    assert not report["passed"]


def test_srm_detected(hillstrom_df: pd.DataFrame) -> None:
    control = hillstrom_df[hillstrom_df["segment"] == "No E-Mail"]
    skewed = pd.concat([hillstrom_df, control], ignore_index=True)  # control doubled
    report = run_checks(skewed, _spec_for(skewed))
    assert not report["checks"]["srm"]["passed"]


def test_leakage_guard() -> None:
    with pytest.raises(LeakageError, match="exposure"):
        assert_no_leakage([*CRITEO_SPEC.features, "exposure"], CRITEO_SPEC)
    with pytest.raises(LeakageError, match="treatment"):
        assert_no_leakage(["recency", "treatment"], HILLSTROM_SPEC)
    assert_no_leakage(HILLSTROM_SPEC.features, HILLSTROM_SPEC)
    assert_no_leakage(CRITEO_SPEC.features, CRITEO_SPEC)


def test_validate_writes_report_and_raises(hillstrom_df: pd.DataFrame, tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    hillstrom_df.drop(columns="treatment").to_parquet(raw / "hillstrom.parquet")
    # Only 3k rows vs the real 64k: the row_count hard check must fail.
    with pytest.raises(DataValidationError, match="row_count"):
        validate("hillstrom", raw, tmp_path / "reports")
    report = json.loads((tmp_path / "reports" / "validation" / "hillstrom.json").read_text())
    assert report["checks"]["row_count"]["observed"] == len(hillstrom_df)
