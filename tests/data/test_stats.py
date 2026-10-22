import numpy as np
import pandas as pd
import pytest

from uplift.data.stats import (
    average_treatment_effect,
    covariate_balance,
    srm_test,
    standardized_mean_difference,
)


def test_ate_recovers_true_effect() -> None:
    rng = np.random.default_rng(1)
    t = rng.integers(0, 2, 200_000)
    y = (rng.random(t.size) < 0.10 + 0.05 * t).astype(int)
    est = average_treatment_effect(y, t)
    assert est.ci_low < 0.05 < est.ci_high
    assert est.p_value < 1e-6
    assert est.relative_lift == pytest.approx(0.5, abs=0.05)


def test_ate_no_effect_is_not_significant() -> None:
    rng = np.random.default_rng(2)
    t = rng.integers(0, 2, 50_000)
    y = rng.random(t.size) < 0.1
    est = average_treatment_effect(y, t)
    assert est.ci_low < 0 < est.ci_high


def test_ate_zero_variance() -> None:
    est = average_treatment_effect([1, 1, 0, 0], [1, 1, 0, 0])
    assert est.ate == 1.0
    assert est.se == 0.0
    assert est.p_value == 0.0


def test_ate_input_errors() -> None:
    with pytest.raises(ValueError, match="same shape"):
        average_treatment_effect([1, 0, 1], [1, 0])
    with pytest.raises(ValueError, match="two units"):
        average_treatment_effect([1, 0, 1], [1, 1, 0])


def test_srm_passes_on_design_ratio() -> None:
    assert srm_test({"t": 8_500, "c": 1_500}, {"t": 0.85, "c": 0.15}).passed


def test_srm_flags_mismatch() -> None:
    result = srm_test({"t": 5_300, "c": 4_700}, {"t": 0.5, "c": 0.5})
    assert not result.passed
    assert result.p_value < 1e-6


def test_srm_validates_arms() -> None:
    with pytest.raises(ValueError, match="arms differ"):
        srm_test({"a": 1, "b": 1}, {"a": 0.5, "c": 0.5})
    with pytest.raises(ValueError, match="sum to 1"):
        srm_test({"a": 1, "b": 1}, {"a": 0.5, "b": 0.6})


def test_smd() -> None:
    rng = np.random.default_rng(3)
    t = np.repeat([0, 1], 5_000)
    same = rng.normal(size=t.size)
    shifted = same + 0.5 * t
    assert abs(standardized_mean_difference(same, t)) < 0.1
    assert standardized_mean_difference(shifted, t) == pytest.approx(0.5, abs=0.06)
    assert standardized_mean_difference(np.ones(t.size), t) == 0.0
    assert standardized_mean_difference(t, t) == np.inf  # arms fully separated


def test_covariate_balance_expands_categoricals() -> None:
    df = pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0] * 50,
            "c": pd.Categorical(["a", "b"] * 100),
            "treatment": [0, 1] * 100,
        }
    )
    out = covariate_balance(df, ["x", "c"], "treatment")
    assert set(out["feature"]) == {"x", "c=a", "c=b"}
    # c is perfectly confounded with treatment here, so it must be flagged.
    assert not out.loc[out["feature"] == "c=a", "balanced"].item()
