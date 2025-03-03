from collections import namedtuple
import functools
import polars as pl
import numpy as np
import altair as alt
import pwlf
from scipy.optimize import curve_fit

from weather.helpers.filter import init_df
from .format_check import is_x_formatted_correcly


def peak_temp_prepare(b: float):
    """
    k: roundness, k=1 is max roundness [0.1, 1] \n
    b: max temperature ~ y-intercept of sorts
    x: numpy array that is symmetric about 0
    """

    q = 0.01  # adjustment for the quadratic function
    a = 1  # adjustment for the absolute value function

    def peak_temp_fit(x, k):
        # assert k > 0 and k <= 1
        assert is_x_formatted_correcly(x)

        quadratic_term = -(k * q * (x**2))
        abs_val_term = -abs((1 - k) * a * x)

        return quadratic_term + abs_val_term + b

    return peak_temp_fit


def get_max_temp_and_time(df: pl.DataFrame, day: int):
    day_df = df.with_columns(day=pl.col("datetime").dt.day()).filter(
        pl.col("day") == day
    )
    res = (
        day_df[day_df["Dry Bulb Temperature"].arg_max()]
        .select(
            pl.col("datetime").dt.hour().alias("hour"),
            pl.col("Dry Bulb Temperature").alias("temp"),
        )
        .to_dicts()[0]
    )
    return res.values()


def generate_xs(extent: int):
    assert extent > 0
    n = extent + 1
    x0 = np.sort(np.arange(start=-1, stop=-n, step=-1))
    x1 = np.arange(n)
    return np.hstack([x0, x1])


TrueData = namedtuple("TrueData", ["hours", "xs", "ys"])
FitResult = namedtuple("FitResult", ["func", "popt", "pcov"])
PeakValues = namedtuple("PeakValues", ["peak_temp", "peak_hour"])


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


def prepare_func(df: pl.DataFrame, day: int, extent=9):
    peak_hour, peak_temp = get_max_temp_and_time(df, day)
    true_data = get_data_to_fit(df, extent, peak_hour, day)
    # func = partial(peak_temp_fit, b=peak_temp)
    func = peak_temp_prepare(b=peak_temp)
    return func, true_data, PeakValues(peak_temp, peak_hour)


def fit_func(df: pl.DataFrame, day: int, bounds=[0.01, 1]):
    func, true_data, peak_values = prepare_func(df, day)
    popt, pcov = curve_fit(func, true_data.xs, true_data.ys, bounds=bounds)
    # print(popt, pcov)
    return true_data, FitResult(func, popt, pcov), peak_values


@functools.lru_cache
def prepare_many_fits(bounds=[1e-3, 1], n_days=9):
    print(bounds)

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
                "peak_temp": [peak_values.peak_temp] * data_len,
                "peak_hour": [peak_values.peak_hour] * data_len,
                "formatted_fit": [formatted_fit] + [""] * (data_len - 1),
            }
        )

    month_df = init_df()
    dfs = [create_day_df(i) for i in range(1, n_days + 1)]
    return pl.concat(dfs)


def plot_many_fits(fits_df: pl.DataFrame):
    base = (
        alt.Chart(fits_df)
        .mark_line()
        .encode(alt.X("hours(time):T", title="hours"), alt.Y("ys").scale(zero=False))
        .properties(width=100, height=100)
    )

    fit = base.mark_line().encode(
        alt.X("hours(time):T", title="hours"),
        alt.Y("fit_ys").scale(zero=False),
        color=alt.value("red"),
        strokeDash=alt.value([2, 2]),
    )

    text1 = base.mark_text(baseline="top", align="left").encode(
        y=alt.value(90),
        x=alt.value(12),
        # all the same value on a given day, so mean is taking the unique
        text=alt.Text("formatted_fit"),
    )

    res = (base + fit + text1).facet(facet=alt.Column("day"), columns=4)

    return res


def create_regression_model(fits_df: pl.DataFrame):
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
    # TO PREDICT: `model.predict(grouped_fit["peak_temp"])`
    return model

    # source = grouped_fit.with_columns(
    # fit_popt=pl.Series(predicted))


# def plot_regression_model():
#     pass
