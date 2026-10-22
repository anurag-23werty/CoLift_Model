import numpy as np

from uplift.logging_utils import get_logger
from uplift.utils import set_seed


def test_set_seed_is_reproducible() -> None:
    set_seed(7)
    a = np.random.rand(5)
    set_seed(7)
    b = np.random.rand(5)
    np.testing.assert_array_equal(a, b)


def test_get_logger_is_idempotent() -> None:
    first = get_logger("uplift.test")
    second = get_logger("uplift.test")
    assert first is second
    assert len(second.handlers) == 1
