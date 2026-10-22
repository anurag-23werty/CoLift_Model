import numpy as np
import pandas as pd
import pytest

from uplift.data.datasets import HILLSTROM_SPEC, add_treatment


def make_hillstrom(n: int = 3_000, seed: int = 0) -> pd.DataFrame:
    """Synthetic data with Hillstrom's schema and a real (positive) email effect."""
    rng = np.random.default_rng(seed)
    segment = rng.choice(["Mens E-Mail", "Womens E-Mail", "No E-Mail"], size=n)
    treated = segment != "No E-Mail"
    visit = (rng.random(n) < np.where(treated, 0.17, 0.10)).astype("int8")
    conversion = (visit & (rng.random(n) < 0.08)).astype("int8")
    spend = np.where(conversion == 1, rng.uniform(30, 500, n), 0.0).astype("float32")
    df = pd.DataFrame(
        {
            "recency": rng.integers(1, 13, n).astype("int16"),
            "history": rng.gamma(2.0, 120.0, n).astype("float32"),
            "mens": rng.integers(0, 2, n).astype("int8"),
            "womens": rng.integers(0, 2, n).astype("int8"),
            "zip_code": pd.Categorical(rng.choice(["Urban", "Surburban", "Rural"], n)),
            "newbie": rng.integers(0, 2, n).astype("int8"),
            "channel": pd.Categorical(rng.choice(["Web", "Phone", "Multichannel"], n)),
            "segment": pd.Categorical(segment),
            "visit": visit,
            "conversion": conversion,
            "spend": spend,
        }
    )
    return add_treatment(df, HILLSTROM_SPEC)


@pytest.fixture
def hillstrom_df() -> pd.DataFrame:
    return make_hillstrom()
