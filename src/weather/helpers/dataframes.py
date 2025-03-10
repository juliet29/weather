import functools
from functools import partial
import polars as pl
from datetime import datetime

from weather.helpers.epw_read import read_epw
from weather.helpers.weather_data import PALO_ALTO_20

from .weather_data import WeatherData

TEMPERATURE = "Dry Bulb Temperature"


def filter_df_by_month(df: pl.DataFrame, weather_data: WeatherData, month, n_days=30):
    assert month != 2 and n_days > 28
    return df.filter(
        pl.col("datetime").is_between(
            datetime(weather_data.start_year, month, 1),
            datetime(weather_data.start_year, month, n_days),
        )
    )


# @functools.lru_cache()
def init_month_df(month=6, weather_data=PALO_ALTO_20):
    """
    :param month: month of the year
    :param weather_data: object of type WeatherData
    :returns: Dataframe with the following schema: [('datetime', Datetime()), ('Dry Bulb Temperature', Float64)]
    """
    assert weather_data.dtype == "EPW"
    df = read_epw(weather_data.path)
    month_filter = partial(filter_df_by_month, df, weather_data)
    mdf = (
        month_filter(month)
        .filter(pl.col("datetime").dt.day() != 30)
        .select(["datetime", TEMPERATURE])
    )  # last day has only 23 values intead of 24..
    assert (mdf["datetime"].dt.date().unique_counts().unique() == 24).all()
    return mdf


def filter_by_day(df: pl.DataFrame, day: int):
    assert 1 <= day <= 31
    return df.filter(pl.col("datetime").dt.day() == day)


def filter_by_days(df: pl.DataFrame, days: list[int] | range):
    for day in days:
        assert 1 <= day <= 31
    return df.filter(pl.col("datetime").dt.day().is_in(days))
