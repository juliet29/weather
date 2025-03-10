import functools

import polars as pl
import altair as alt
import pwlf
from scipy.optimize import curve_fit

from weather.helpers.time import create_centered_interval
from weather.helpers.dataframes import init_month_df
from ..interfaces import Profile

from .calc import calc_peak_profile, peak_temp_prepare
from .helpers import (
    EXTENT,
    FitResult,
    PeakValues,
    TrueData,
    generate_xs,
    get_max_temp_and_time,
)


def get_data_to_fit(df: pl.DataFrame, extent: int, peak_hour: int, day: int):
    """
    df: Schema([('datetime', Datetime(time_unit='us', time_zone=None)),
        ('Dry Bulb Temperature', Float64)])
    day: 1 - 30* (may be less than 30 days)!
    """

    min_hour = peak_hour - extent
    max_hour = peak_hour + extent
    assert min_hour >= 0 and max_hour <= 23

    xs = generate_xs(extent)

    extent_df = df.with_columns(
        day=pl.col("datetime").dt.day(), hour=pl.col("datetime").dt.hour()
    ).filter((pl.col("hour") >= min_hour) & (pl.col("hour") <= max_hour))

    assert day in extent_df["day"].unique()
    ys = extent_df.filter(pl.col("day") == day)["Dry Bulb Temperature"].to_numpy()
    hours = extent_df.filter(pl.col("day") == day)["datetime"].to_list()

    return TrueData(hours, xs, ys)


def prepare_func(df: pl.DataFrame, day: int, extent=EXTENT):
    peak_values = get_max_temp_and_time(df, day)
    true_data = get_data_to_fit(df, extent, peak_values.hour, day)
    # func = partial(peak_temp_fit, b=peak_temp)
    func = peak_temp_prepare(b=peak_values.temp)
    return func, true_data, peak_values


def fit_func(df: pl.DataFrame, day: int, bounds=[0.01, 1]):
    func, true_data, peak_values = prepare_func(df, day)
    popt, pcov = curve_fit(func, true_data.xs, true_data.ys, bounds=bounds)
    return true_data, FitResult(func, popt, pcov), peak_values


@functools.lru_cache
def prepare_many_fits(bounds=[1e-3, 1], n_days=9):

    def create_day_df(day: int):
        true_data, fit_result, peak_values = fit_func(month_df, day, bounds)
        data_len = len(true_data.hours)

        formatted_fit = f"r={fit_result.popt[0]:.3f}({fit_result.pcov[0][0]:.1E})"

        return pl.DataFrame(
            {
                "day": [day] * data_len,
                "time": true_data.hours,
                "xs": true_data.xs,
                "ys": true_data.ys,
                "fit_ys": fit_result.func(true_data.xs, *fit_result.popt),
                "popt": [fit_result.popt[0]] * data_len,
                "pcov": [fit_result.pcov[0][0]] * data_len,
                "peak_temp": [peak_values.temp] * data_len,
                "peak_hour": [peak_values.hour] * data_len,
                "formatted_fit": [formatted_fit] + [""] * (data_len - 1),
            }
        )

    month_df = init_month_df()
    dfs = [create_day_df(i) for i in range(1, n_days + 1)]
    return pl.concat(dfs)


def create_r_peak_temp_model(fits_df: pl.DataFrame):
    def plot_fit():
        predicted = model.predict(grouped_fit["peak_temp"])
        source = grouped_fit.with_columns(fit_popt=pl.Series(predicted))

        base = (
            alt.Chart(source)
            .mark_circle()
            .encode(
                alt.X("peak_temp:Q").scale(zero=False),
                alt.Y("popt:Q").scale(zero=False),
            )
        )

        fit_pwlf = base.mark_line().encode(
            alt.X("peak_temp:Q").scale(zero=False),
            alt.Y("fit_popt:Q").scale(zero=False),
            color=alt.value("black"),
        )

        print(f"P-values: {model.p_values()}")

        return (base + fit_pwlf).show()

    grouped_fit = fits_df.group_by("day").agg(
        [
            pl.col("popt").unique().first(),
            pl.col("peak_temp").unique().first(),
            pl.col("peak_hour").unique().first(),
        ]
    )

    model = pwlf.PiecewiseLinFit(grouped_fit["peak_temp"], grouped_fit["popt"])
    # model has two parameters on observation
    model.fit(2)

    # plot_fit()
    return model


def fit_peak_profile(
    df: pl.DataFrame, day: int, r_peak_temp_model: pwlf.PiecewiseLinFit, extent=EXTENT
):
    peak_values = get_max_temp_and_time(df, day)
    # print(f"peak_values: {peak_values}")

    r = r_peak_temp_model.predict(peak_values.temp)[0]
    # print(f"r for peak_fit: {r}")
    xs = generate_xs(extent)

    return Profile(
        calc_peak_profile(xs, peak_values.temp, r),
        create_centered_interval(int(peak_values.hour)),
    )
