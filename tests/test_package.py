import importlib

import pytest

import uplift

SUBPACKAGES = [
    "data",
    "features",
    "models",
    "evaluation",
    "policy",
    "serving",
    "monitoring",
]


def test_version_is_set() -> None:
    assert isinstance(uplift.__version__, str)
    assert uplift.__version__ != ""


@pytest.mark.parametrize("name", SUBPACKAGES)
def test_subpackages_importable(name: str) -> None:
    importlib.import_module(f"uplift.{name}")
