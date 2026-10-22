"""Download the raw experiment datasets and convert them to typed parquet.

Usage:
    python -m uplift.data.download --dataset all
    python -m uplift.data.download --dataset criteo --force
"""

from __future__ import annotations

import argparse
import hashlib
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from uplift.config import load_config
from uplift.logging_utils import get_logger

log = get_logger(__name__)

_CHUNK = 1 << 20  # 1 MiB


@dataclass(frozen=True)
class DatasetSource:
    name: str
    url: str
    sha256: str | None
    dtypes: dict[str, str]

    @property
    def archive_name(self) -> str:
        return f"{self.name}.csv.gz"

    @property
    def parquet_name(self) -> str:
        return f"{self.name}.parquet"


HILLSTROM = DatasetSource(
    name="hillstrom",
    url="https://hillstorm1.s3.us-east-2.amazonaws.com/hillstorm_no_indices.csv.gz",
    sha256="bab6578f60db5d792f1c2372c502f029152a5249cf5ea84390f3b7f885d7234f",
    dtypes={
        "recency": "int16",
        "history_segment": "category",
        "history": "float32",
        "mens": "int8",
        "womens": "int8",
        "zip_code": "category",
        "newbie": "int8",
        "channel": "category",
        "segment": "category",
        "visit": "int8",
        "conversion": "int8",
        "spend": "float32",
    },
)

CRITEO = DatasetSource(
    name="criteo",
    # Official Criteo AI Lab mirror (v2.1, CC BY-NC-SA 4.0); the old S3 bucket now returns 404.
    url=(
        "https://huggingface.co/datasets/criteo/criteo-uplift/resolve/main/"
        "criteo-research-uplift-v2.1.csv.gz"
    ),
    sha256="2716e1bf0fd157a93b5bf86924d9088419dfbac2022c6cd90030220634f616dc",
    dtypes={
        **{f"f{i}": "float32" for i in range(12)},
        "treatment": "int8",
        "conversion": "int8",
        "visit": "int8",
        "exposure": "int8",
    },
)

SOURCES: dict[str, DatasetSource] = {s.name: s for s in (HILLSTROM, CRITEO)}


def sha256sum(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def download_archive(source: DatasetSource, raw_dir: Path, force: bool = False) -> Path:
    """Stream the compressed CSV to `raw_dir`, verifying its checksum when known."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    dest = raw_dir / source.archive_name
    if dest.exists() and not force:
        log.info("%s already downloaded at %s", source.name, dest)
    else:
        tmp = dest.with_suffix(dest.suffix + ".part")
        log.info("Downloading %s from %s", source.name, source.url)
        with urllib.request.urlopen(source.url) as resp, tmp.open("wb") as out:
            total = int(resp.headers.get("Content-Length", 0))
            done, next_report = 0, 0.1
            while chunk := resp.read(_CHUNK):
                out.write(chunk)
                done += len(chunk)
                if total and done / total >= next_report:
                    pct, mb = 100 * done / total, total / 1e6
                    log.info("  %s: %3.0f%% of %.0f MB", source.name, pct, mb)
                    next_report += 0.1
        tmp.replace(dest)

    digest = sha256sum(dest)
    if source.sha256 and digest != source.sha256:
        raise ValueError(
            f"Checksum mismatch for {dest}: expected {source.sha256}, got {digest}. "
            "Delete the file and re-run with --force."
        )
    log.info("%s sha256=%s", source.name, digest)
    return dest


def to_parquet(source: DatasetSource, archive: Path, raw_dir: Path) -> Path:
    """Read the CSV with explicit dtypes (halves memory for Criteo) and write parquet."""
    dest = raw_dir / source.parquet_name
    log.info("Converting %s to parquet", archive.name)
    df = pd.read_csv(archive, dtype=source.dtypes)
    df.to_parquet(dest, index=False)
    log.info("Wrote %s: %d rows x %d cols", dest, len(df), df.shape[1])
    return dest


def fetch(name: str, raw_dir: Path, force: bool = False) -> Path:
    source = SOURCES[name]
    parquet = raw_dir / source.parquet_name
    if parquet.exists() and not force:
        log.info("%s parquet exists at %s, skipping", name, parquet)
        return parquet
    archive = download_archive(source, raw_dir, force=force)
    return to_parquet(source, archive, raw_dir)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=[*SOURCES, "all"], default="all")
    parser.add_argument("--config", default=None, help="path to YAML config")
    parser.add_argument("--force", action="store_true", help="re-download and re-convert")
    args = parser.parse_args(argv)

    cfg = load_config(args.config) if args.config else load_config()
    names = list(SOURCES) if args.dataset == "all" else [args.dataset]
    for name in names:
        fetch(name, cfg.paths.raw_dir, force=args.force)


if __name__ == "__main__":
    main()

__all__ = ["SOURCES", "DatasetSource", "download_archive", "fetch", "sha256sum", "to_parquet"]
