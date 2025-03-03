from functools import partial
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from .format_check import is_x_formatted_correcly

from weather.helpers.epw_read import read_epw
from weather.helpers.weather_data import PALO_ALTO_20
from weather.helpers.filter import filter_df_by_month

def init_df(month=6):
    df = read_epw(PALO_ALTO_20.path)
    month_filter = partial(filter_df_by_month, df, PALO_ALTO_20)
    mdf = (
        month_filter(month)
        .filter(pl.col("datetime").dt.day() != 30)
        .select(["datetime", "Dry Bulb Temperature"])
    )  # last day has only 23 values intead of 24..
    assert (mdf["datetime"].dt.date().unique_counts().unique() == 24).all()
    return mdf


def peak_temp_prepare(b: float):
    """
    k: roundness, k=1 is max roundness [0.1, 1] \n
    b: max temperature ~ y-intercept of sorts
    x: numpy array that is symmetric about 0
    """

    q = 0.01  # adjustment for the quadratic function
    a = 1 # adjustment for the absolute value function


    def peak_temp_fit(x, k):
        # assert k > 0 and k <= 1
        assert is_x_formatted_correcly(x)

        quadratic_term = -(k * q * (x**2))
        abs_val_term =  -abs((1-k)*a*x)

        return quadratic_term + abs_val_term + b
    
    return peak_temp_fit

# def peak_temp_prepare(b):
#     q = 0.1
#     return lambda x, k: -(k * q * (x**2)) + b


# TODO - find k through some opt scheme..


def get_max_temp_and_time(df: pl.DataFrame, day: int):
    day_df = df.with_columns(day=pl.col("datetime").dt.day()).filter(
        pl.col("day") == day
    )
    res =  (
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

    return (hours, xs, ys)


def prepare_func(df: pl.DataFrame, day: int, extent=9):

    peak_hour, peak_temp = get_max_temp_and_time(df, day)
    hours, xs, ys = get_data_to_fit(df, extent, peak_hour, day)
    # func = partial(peak_temp_fit, b=peak_temp)
    func = peak_temp_prepare(b=peak_temp)
    return func, xs, ys


def fit_func(df: pl.DataFrame, day: int, bounds=[0.01,1]):
    func, xs, ys  = prepare_func(df, day)
    popt, pcov = curve_fit(func, xs, ys, bounds=bounds)
    print(popt, pcov)
    plt.plot(xs, ys, 'b-', label='data')
    plt.plot(xs, func(xs, *popt), 'r-', label=f'fit: {popt[0]:.3f}, bound=({bounds[0]:.3f}, {bounds[1]:.3f})')
    plt.title(f"Day {day}")
    plt.legend()
    plt.show()


    # # extent is 9, incase peak is at one.. 
    # # but plot the times of max also.. 
    # pass

