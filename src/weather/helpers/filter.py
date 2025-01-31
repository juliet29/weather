import polars as pl
from datetime import datetime

from .weather_data import WeatherData


def filter_df_by_month(df: pl.DataFrame, weather_data: WeatherData, month, n_days=30):
    assert month != 2 and n_days > 28
    return df.filter(
        pl.col("datetime").is_between(
            datetime(weather_data.year, month, 1),
            datetime(weather_data.year, month, n_days),
        )
    )
