from collections import namedtuple
from typing import NamedTuple
import numpy as np
from ...helpers.dataframes import TEMPERATURE

import polars as pl

PeakValues = NamedTuple("PeakValues", [("hour", int), ("temp", float)])
TrueData = namedtuple("TrueData", ["hours", "xs", "ys"])
FitResult = namedtuple("FitResult", ["func", "popt", "pcov"])
EXTENT = 9


def get_max_temp_and_time(df: pl.DataFrame, day: int):
    day_df = df.with_columns(day=pl.col("datetime").dt.day()).filter(
        pl.col("day") == day
    )
    res = (
        day_df[day_df[TEMPERATURE].arg_max()] # type: ignore
        .select(
            pl.col("datetime").dt.hour().alias("hour"),
            pl.col(TEMPERATURE).alias("temp"),
        )
        .to_dicts()[0]
    )
    return PeakValues(*res.values())


def generate_xs(extent: int):
    assert extent > 0
    n = extent + 1
    x0 = np.sort(np.arange(start=-1, stop=-n, step=-1))
    x1 = np.arange(n)
    return np.hstack([x0, x1])
