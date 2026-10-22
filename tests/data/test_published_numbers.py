"""Checks against the real downloaded data. Skipped when the data is not present (e.g. in CI)."""

import pytest

from uplift.config import load_config
from uplift.data.datasets import CRITEO_SPEC, HILLSTROM_SPEC, load_raw
from uplift.data.stats import average_treatment_effect

RAW = load_config().paths.raw_dir

requires_hillstrom = pytest.mark.skipif(
    not (RAW / "hillstrom.parquet").exists(), reason="Hillstrom not downloaded"
)
requires_criteo = pytest.mark.skipif(
    not (RAW / "criteo.parquet").exists(), reason="Criteo not downloaded"
)

# Per-arm lift reported by Kevin Hillstrom (MineThatData, 2008).
HILLSTROM_PUBLISHED = {
    ("Mens E-Mail", "visit"): 0.0766,
    ("Womens E-Mail", "visit"): 0.0452,
    ("Mens E-Mail", "conversion"): 0.0068,
    ("Womens E-Mail", "conversion"): 0.0031,
}


@requires_hillstrom
@pytest.mark.integration
@pytest.mark.parametrize(("arm", "outcome"), list(HILLSTROM_PUBLISHED))
def test_hillstrom_matches_published_lift(arm: str, outcome: str) -> None:
    df = load_raw("hillstrom", RAW)
    df = df[df["segment"].isin([arm, HILLSTROM_SPEC.control_arm])]
    est = average_treatment_effect(df[outcome], df["segment"] == arm)
    assert est.ate == pytest.approx(HILLSTROM_PUBLISHED[(arm, outcome)], abs=5e-4)


@requires_criteo
@pytest.mark.integration
@pytest.mark.slow
def test_criteo_matches_published_rates() -> None:
    df = load_raw("criteo", RAW)
    assert len(df) == CRITEO_SPEC.expected_rows
    # Diemert et al. (2018), v2.1: 85% treated, ~4.7% visit, ~0.29% conversion.
    assert df["treatment"].mean() == pytest.approx(0.85, abs=0.001)
    assert df["visit"].mean() == pytest.approx(0.047, abs=0.001)
    assert df["conversion"].mean() == pytest.approx(0.0029, abs=0.0002)
