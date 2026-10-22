import pandas as pd
import pytest

from uplift.config import Config
from uplift.data.datasets import HILLSTROM_SPEC
from uplift.data.split import (
    ROW_ID,
    frame_fingerprint,
    make_splits,
    split_frame,
    strata,
    stratified_sample,
)

from .conftest import make_hillstrom


@pytest.fixture
def df() -> pd.DataFrame:
    out = make_hillstrom(n=6_000, seed=5)
    out.insert(0, ROW_ID, range(len(out)))
    return out


def test_split_sizes_and_disjoint(df: pd.DataFrame) -> None:
    parts = split_frame(df, HILLSTROM_SPEC, val_size=0.15, test_size=0.15, seed=0)
    sizes = {k: len(v) for k, v in parts.items()}
    assert sum(sizes.values()) == len(df)
    assert sizes["test"] == pytest.approx(0.15 * len(df), abs=2)
    assert sizes["val"] == pytest.approx(0.15 * len(df), abs=2)
    ids = [set(p[ROW_ID]) for p in parts.values()]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])


def test_split_preserves_strata(df: pd.DataFrame) -> None:
    parts = split_frame(df, HILLSTROM_SPEC, val_size=0.15, test_size=0.15, seed=0)
    overall = strata(df, HILLSTROM_SPEC).value_counts(normalize=True)
    for part in parts.values():
        share = strata(part, HILLSTROM_SPEC).value_counts(normalize=True)
        pd.testing.assert_series_equal(
            share.sort_index(), overall.sort_index(), atol=0.01, check_names=False
        )


def test_split_is_deterministic(df: pd.DataFrame) -> None:
    def split_hash(seed: int) -> str:
        return frame_fingerprint(split_frame(df, HILLSTROM_SPEC, 0.15, 0.15, seed)["test"])

    assert split_hash(0) == split_hash(0)
    assert split_hash(0) != split_hash(1)


def test_stratified_sample_keeps_rates(df: pd.DataFrame) -> None:
    sample = stratified_sample(df, HILLSTROM_SPEC, frac=0.5, seed=0)
    assert len(sample) == pytest.approx(len(df) / 2, abs=2)
    assert sample["conversion"].mean() == pytest.approx(df["conversion"].mean(), abs=0.002)
    assert stratified_sample(df, HILLSTROM_SPEC, frac=1.0, seed=0) is df


def test_make_splits_end_to_end(tmp_path, df: pd.DataFrame) -> None:  # type: ignore[no-untyped-def]
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    df.drop(columns=[ROW_ID, "treatment"]).to_parquet(raw / "hillstrom.parquet")
    cfg = Config.model_validate(
        {
            "paths": {
                "raw_dir": raw,
                "processed_dir": tmp_path / "data" / "processed",
                "reports_dir": tmp_path / "reports",
            }
        }
    )
    first = make_splits("hillstrom", cfg)
    second = make_splits("hillstrom", cfg)
    for split in ("train", "val", "test"):
        assert first[split]["sha256"] == second[split]["sha256"]
        assert (tmp_path / "data" / "processed" / "hillstrom" / f"{split}.parquet").exists()
    assert (tmp_path / "reports" / "splits" / "hillstrom.json").exists()
