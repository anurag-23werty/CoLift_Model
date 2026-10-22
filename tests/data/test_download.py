from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from uplift.data.download import HILLSTROM, DatasetSource, download_archive, sha256sum, to_parquet


@pytest.fixture
def local_source(tmp_path: Path) -> DatasetSource:
    """A tiny gzipped CSV served over file:// so tests never hit the network."""
    src_dir = tmp_path / "remote"
    src_dir.mkdir()
    archive = src_dir / "tiny.csv.gz"
    pd.DataFrame({"x": [1.5, 2.5, 3.5], "treatment": [0, 1, 1]}).to_csv(archive, index=False)
    return DatasetSource(
        name="tiny",
        url=archive.as_uri(),
        sha256=sha256sum(archive),
        dtypes={"x": "float32", "treatment": "int8"},
    )


def test_download_and_convert(local_source: DatasetSource, tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    archive = download_archive(local_source, raw)
    assert archive == raw / "tiny.csv.gz"
    assert not archive.with_suffix(".gz.part").exists()

    df = pd.read_parquet(to_parquet(local_source, archive, raw))
    assert len(df) == 3
    assert df["x"].dtype == "float32"
    assert df["treatment"].dtype == "int8"


def test_checksum_mismatch_raises(local_source: DatasetSource, tmp_path: Path) -> None:
    bad = replace(local_source, sha256="0" * 64)
    with pytest.raises(ValueError, match="Checksum mismatch"):
        download_archive(bad, tmp_path / "raw")


def test_existing_archive_not_redownloaded(local_source: DatasetSource, tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    first = download_archive(local_source, raw)
    mtime = first.stat().st_mtime_ns
    download_archive(local_source, raw)
    assert first.stat().st_mtime_ns == mtime


def test_hillstrom_dtypes_cover_all_columns() -> None:
    expected = {
        "recency", "history_segment", "history", "mens", "womens", "zip_code",
        "newbie", "channel", "segment", "visit", "conversion", "spend",
    }  # fmt: skip
    assert set(HILLSTROM.dtypes) == expected
