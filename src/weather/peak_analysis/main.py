import altair as alt
import polars as pl
import pendulum
from ..helpers.dataframes import filter_by_days, init_month_df, TEMPERATURE
from ..fit.day_fit import MORNING, EVENING


def create_analysis_df(n_days=12):
    return filter_by_days(init_month_df(), range(1, n_days + 1))


def plot_temps_of_many_days(df: pl.DataFrame):
    base = (
        alt.Chart(df)
        .mark_line()
        .encode(alt.X("hours(datetime)"), alt.Y(TEMPERATURE), color=alt.value("grey"))
        .properties(width=100, height=100)
    )

    return (base).facet("date(datetime)", columns=4)


def plot_values_of_many_days(df: pl.DataFrame, value):
    base = (
        alt.Chart(df)
        .mark_line()
        .encode(alt.X("hours(datetime)"), alt.Y(value), color=alt.value("grey"))
        .properties(width=100, height=100)
    )

    return (base).facet("date(datetime)", columns=4)


# now want to see afternoon derivatives.. for all days..
# filter to desired time..
# then calc deriv here..
def create_deriv_df(_df: pl.DataFrame):
    df = (
        _df.filter(
            pl.col("datetime").dt.hour().is_between(MORNING.hour_2, EVENING.hour_1)
        )
        .with_columns(deriv=pl.col(TEMPERATURE).diff().fill_null(strategy="zero"))
        .with_columns(deriv2=pl.col("deriv").diff().fill_null(strategy="zero"))
    )

    return df

    # max_lag = data.shape[0] - 1 if max_lag is None else int(max_lag)
    # lags = np.arange(0, max_lag + 1)
    # _data = pd.DataFrame(
    #     dict(Lag=lags, Autocorrelation=[data[column].autocorr(lag=lag) for lag in lags])
    # )
    # return (
    #     alt.Chart(_data, height=height, width=width)
    #     .mark_bar()
    #     .encode(x="Lag:O", y="Autocorrelation")
    # )
