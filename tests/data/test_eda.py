import matplotlib

matplotlib.use("Agg")

import pandas as pd

from uplift.data.datasets import HILLSTROM_SPEC
from uplift.data.eda import effect_by_segment, outcome_rates_by_arm
from uplift.data.stats import covariate_balance
from uplift.viz import apply_style, plot_balance, plot_effect_forest, plot_rates_by_arm


def test_outcome_rates_by_arm(hillstrom_df: pd.DataFrame) -> None:
    rates = outcome_rates_by_arm(hillstrom_df, HILLSTROM_SPEC)
    assert set(rates.index) == set(HILLSTROM_SPEC.arm_ratios)
    assert rates["n"].sum() == len(hillstrom_df)
    visit = rates["visit"]
    assert float(visit["Mens E-Mail"]) > float(visit["No E-Mail"])


def test_effect_by_segment_shape(hillstrom_df: pd.DataFrame) -> None:
    out = effect_by_segment(hillstrom_df, HILLSTROM_SPEC, ["channel", "newbie"], "visit")
    # 3 channel levels + 2 newbie levels, each for 2 treated arms
    assert len(out) == (3 + 2) * 2
    assert (out["ci_low"] <= out["ate"]).all()
    assert (out["ate"] <= out["ci_high"]).all()
    assert set(out["arm"]) == {"Mens E-Mail", "Womens E-Mail"}


def test_plots_render(hillstrom_df: pd.DataFrame) -> None:
    apply_style()
    rates = outcome_rates_by_arm(hillstrom_df, HILLSTROM_SPEC)
    effects = effect_by_segment(hillstrom_df, HILLSTROM_SPEC, ["channel"], "visit")
    balance = covariate_balance(hillstrom_df, HILLSTROM_SPEC.features, "treatment")
    for fig in (
        plot_rates_by_arm(rates, ["visit", "conversion"], "rates"),
        plot_effect_forest(effects, "forest", "lift"),
        plot_balance(balance, "balance"),
    ):
        assert fig.axes
        matplotlib.pyplot.close(fig)
