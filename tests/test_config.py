from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from uplift.config import DEFAULT_CONFIG, PROJECT_ROOT, Config, load_config


def test_base_config_loads() -> None:
    cfg = load_config(DEFAULT_CONFIG)
    assert cfg.data.dataset == "hillstrom"
    assert cfg.policy.margin_per_conversion > 0


def test_paths_are_resolved_to_project_root() -> None:
    cfg = load_config(DEFAULT_CONFIG)
    assert cfg.paths.raw_dir == PROJECT_ROOT / "data" / "raw"
    assert all(p.is_absolute() for _, p in cfg.paths)


def test_unknown_keys_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump({"seed": 1, "typo_key": True}))
    with pytest.raises(ValidationError):
        load_config(bad)


def test_criteo_has_no_spend() -> None:
    with pytest.raises(ValidationError, match="no spend"):
        Config.model_validate({"data": {"dataset": "criteo", "outcome": "spend"}})


def test_negative_cost_rejected() -> None:
    with pytest.raises(ValidationError):
        Config.model_validate({"policy": {"discount_cost": -5}})
