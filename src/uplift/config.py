"""Typed project configuration loaded from YAML.

Every pipeline stage reads its settings through `load_config`, so a run is fully
described by one config file (which also gets logged to MLflow later).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "base.yaml"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PathsConfig(_Strict):
    data_dir: Path = Path("data")
    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    models_dir: Path = Path("models")
    reports_dir: Path = Path("reports")

    def resolve(self, root: Path = PROJECT_ROOT) -> PathsConfig:
        """Return a copy with every relative path anchored at `root`."""
        return PathsConfig(**{name: p if p.is_absolute() else root / p for name, p in self})


class DataConfig(_Strict):
    dataset: Literal["hillstrom", "criteo"] = "hillstrom"
    outcome: Literal["visit", "conversion", "spend"] = "conversion"
    criteo_sample_frac: float = Field(default=0.1, gt=0, le=1)
    test_size: float = Field(default=0.15, gt=0, lt=0.5)
    val_size: float = Field(default=0.15, gt=0, lt=0.5)

    @model_validator(mode="after")
    def _check(self) -> DataConfig:
        if self.dataset == "criteo" and self.outcome == "spend":
            raise ValueError("Criteo has no spend column; use 'visit' or 'conversion'.")
        if self.test_size + self.val_size >= 0.6:
            raise ValueError("test_size + val_size leaves too little training data.")
        return self


class PolicyConfig(_Strict):
    """Business levers for the targeting simulator (currency: INR)."""

    margin_per_conversion: float = Field(default=400.0, gt=0)
    discount_cost: float = Field(default=100.0, ge=0)
    contact_cost: float = Field(default=0.5, ge=0)
    target_fraction: float = Field(default=0.3, gt=0, le=1)
    budget: float | None = Field(default=None, gt=0)


class Config(_Strict):
    seed: int = 42
    paths: PathsConfig = PathsConfig()
    data: DataConfig = DataConfig()
    policy: PolicyConfig = PolicyConfig()


def load_config(path: Path | str = DEFAULT_CONFIG) -> Config:
    """Load and validate a YAML config file."""
    with Path(path).open() as f:
        raw = yaml.safe_load(f) or {}
    cfg = Config.model_validate(raw)
    return cfg.model_copy(update={"paths": cfg.paths.resolve()})
