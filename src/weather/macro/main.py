import polars as pl
import altair as alt
from scipy.signal import find_peaks
from ..helpers.weather_data import STANFORD_10

TEMP_F = "Temp (°F)"


def read_stanford_weather_data(TEMP_ONLY=False):
    df = (
        pl.scan_csv(
            STANFORD_10.path,
            schema_overrides={"Daily Rain (In)": pl.Float32, "Time": pl.String},
        )
        .filter(pl.col("Station") == "Stanford Met Tower")
        .with_columns(
            pl.col("Date").str.to_date("%m/%d/%y"),
            pl.col("Time").str.zfill(4).str.to_time("%H%M"),
        )
        .with_columns(pl.col("Date").dt.combine(pl.col("Time")).alias("datetime"))
    )
    if TEMP_ONLY:
        dft = (
            df.select(["datetime", TEMP_F])
            .with_columns(pl.col(TEMP_F).fill_null(pl.col(TEMP_F).median()))
            .collect()
        )
        assert dft.filter(pl.any_horizontal(pl.all().is_null())).is_empty()

        return dft

    return df.collect()


def filter_and_group(df: pl.DataFrame):
    return (
        df.filter(pl.col("datetime").dt.year() > 2018)
        .filter(pl.col("datetime").dt.month().is_in([6, 7, 8]))
        .group_by(pl.col("datetime").dt.date(), maintain_order=True)
    )


def agg_max(df: pl.DataFrame):
    return filter_and_group(df).agg(pl.col(TEMP_F).max().alias("max_temp"))


def plot_many_summer_temps(df_agg: pl.DataFrame, col_name="max_temp"):
    base = (
        alt.Chart(df_agg)
        .mark_line()
        .encode(alt.X("monthdate(datetime)"), alt.Y(f"{col_name}:Q"))
        .properties(width=400, height=100)
    )
    return base.facet("year(datetime)", columns=1)


def plot_summer_temp_distributions():
    df = filter_and_group(read_stanford_weather_data(True))
    df_median = df.agg(pl.col(TEMP_F).mean().alias("mean")).unpivot(
        on=["mean"], index="datetime"
    )

    base = (
        alt.Chart(df_median)
        .mark_bar(opacity=0.5)
        .encode(
            alt.X("value:Q").bin(maxbins=30),
            alt.Y("count()").stack(None),
        )
        .properties(width=400, height=100, )
        .facet("year(datetime)", columns=2, title="Distributions of Mean Temps")
    ).configure_title(fontSize=20)

    return df_median, base

    base = (
        alt.Chart(df_median)
        .mark_bar()
        .encode(alt.X("monthdate(datetime)"), alt.Y("median_temp:Q"))
        .properties(width=400, height=100)
    )
    return base.facet("year(datetime)", columns=1)

    pass
    # want to see which statistic provides the most info about the day..


def find_and_plot_peaks(df_agg: pl.DataFrame):
    # df_agg = filter_and_agg(df)
    dfa = df_agg.filter(pl.col("datetime").dt.year() == 2024)

    peak_ix, _ = find_peaks(dfa["max_temp"], prominence=10)
    n = len(dfa)

    df_peaks = dfa.with_columns(
        pl.Series([1 if i in peak_ix else 0 for i in range(n)]).alias("peak_indicator")
    ).with_columns(
        pl.when(pl.col("peak_indicator") == 1)
        .then(pl.col("max_temp"))
        .otherwise(None)
        .alias("peak_temp")
    )

    base = (
        alt.Chart(df_peaks)
        .mark_line()
        .encode(
            alt.X("datetime"),
            alt.Y("max_temp:Q"),
        )
        .properties(width=400, height=100)
    )

    peak_marks = base.mark_circle().encode(
        alt.X("datetime"), alt.Y("peak_temp:Q"), color=alt.value("red")
    )

    return base + peak_marks, df_peaks
