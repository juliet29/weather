from typing import Callable, NamedTuple
import polars as pl
import altair as alt
from weather.helpers.filter import init_df
from scipy.stats import linregress

LinRegModel = Callable[[float], float]
HourPair = NamedTuple("HourPair", [("hour_1", int), ("hour_2", int)])
MORNING = HourPair(0,4)
EVENING = HourPair(18, 23)

def linear_regression_prediction(fit)-> LinRegModel:
    def fx(x:float) -> float:
        return fit.slope * x + fit.intercept
    return fx


def filter_to_times_of_day(df: pl.DataFrame, hour_pair):
    hour_1, hour_2 = hour_pair
    assert hour_1 < hour_2
    return (
        df.filter(
            (pl.col("datetime").dt.hour() == hour_1)
            | (pl.col("datetime").dt.hour() == hour_2)
        )
        .with_columns(
            hour=pl.col("datetime").dt.hour(), date=pl.col("datetime").dt.date()
        )
        .pivot(on="hour", index="date", values="Dry Bulb Temperature")
        .with_columns(deltaT=pl.col(str(hour_1)) - pl.col(str(hour_2)))
    )


def create_time_of_day_linear_model(df: pl.DataFrame, hour_pair:HourPair) ->LinRegModel :
    hour_1, hour_2 = hour_pair
    filtered_df = filter_to_times_of_day(df, hour_pair)
    fit =  linregress(filtered_df[str(hour_1)], filtered_df[str(hour_2)])
    print(f"r-value: {fit.rvalue}")
    return linear_regression_prediction(fit)


def visually_check_reg(hour_pair:HourPair, model:LinRegModel):
    hour_1, hour_2 = hour_pair
    filtered_df = filter_to_times_of_day(init_df(), hour_pair)

    fit_res = model(filtered_df[str(hour_1)])
    source = filtered_df.with_columns(fit_data = pl.Series(fit_res))
    base = alt.Chart(source).mark_circle().encode(
        alt.X(f"{hour_1}:Q").scale(zero=False),
        alt.Y(f"{hour_2}:Q").scale(zero=False),
    )

    fit_chart = base.mark_line().encode(
        alt.X(f"{hour_1}:Q").scale(zero=False),
        alt.Y("fit_data:Q").scale(zero=False),
        color=alt.value("black")
    )

    return (base + fit_chart)

def create_morning_and_evening_models(show_reg=False) -> tuple[LinRegModel,LinRegModel]:
    df = init_df()
    morning_model = create_time_of_day_linear_model(df, MORNING)
    evening_model = create_time_of_day_linear_model(df, EVENING)

    if show_reg:
        visually_check_reg(MORNING, morning_model)
        visually_check_reg(EVENING, evening_model)

    return morning_model, evening_model


