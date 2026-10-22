"""Pandera schemas: the data contract each raw dataset must satisfy."""

from __future__ import annotations

import pandas as pd
import pandera.pandas as pa

BINARY = pa.Check.isin([0, 1])


def _binary(dtype: str = "int8") -> pa.Column:
    return pa.Column(dtype, BINARY, nullable=False)


HILLSTROM_SCHEMA = pa.DataFrameSchema(
    {
        "recency": pa.Column("int16", pa.Check.in_range(1, 12), nullable=False),
        "history": pa.Column("float32", pa.Check.ge(0), nullable=False),
        "mens": _binary(),
        "womens": _binary(),
        "zip_code": pa.Column(
            "category", pa.Check.isin(["Urban", "Surburban", "Rural"]), nullable=False
        ),
        "newbie": _binary(),
        "channel": pa.Column(
            "category", pa.Check.isin(["Web", "Phone", "Multichannel"]), nullable=False
        ),
        "segment": pa.Column(
            "category",
            pa.Check.isin(["Mens E-Mail", "Womens E-Mail", "No E-Mail"]),
            nullable=False,
        ),
        "visit": _binary(),
        "conversion": _binary(),
        "spend": pa.Column("float32", pa.Check.ge(0), nullable=False),
        "treatment": _binary(),
    },
    checks=[
        pa.Check(
            lambda df: ~((df["conversion"] == 1) & (df["visit"] == 0)),
            error="a conversion requires a visit",
        ),
        pa.Check(
            lambda df: (df["spend"] > 0) == (df["conversion"] == 1),
            error="spend must be positive exactly when the customer converted",
        ),
    ],
    strict=False,  # history_segment is kept in raw data but not validated
)

CRITEO_SCHEMA = pa.DataFrameSchema(
    {
        **{f"f{i}": pa.Column("float32", nullable=False) for i in range(12)},
        "treatment": _binary(),
        "conversion": _binary(),
        "visit": _binary(),
        "exposure": _binary(),
    },
    checks=[
        pa.Check(
            lambda df: ~((df["treatment"] == 0) & (df["exposure"] == 1)),
            error="control users cannot be exposed to the ad",
        ),
        pa.Check(
            lambda df: ~((df["conversion"] == 1) & (df["visit"] == 0)),
            error="a conversion requires a visit",
        ),
    ],
    strict=True,
)

SCHEMAS: dict[str, pa.DataFrameSchema] = {"hillstrom": HILLSTROM_SCHEMA, "criteo": CRITEO_SCHEMA}


def validate_schema(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Validate lazily so every violation is reported at once, not just the first."""
    return SCHEMAS[name].validate(df, lazy=True)
