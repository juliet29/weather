from typing import NamedTuple

import polars as pl
import altair as alt
import pwlf

from weather.fit.interfaces import Profile
from weather.helpers.dataframes import filter_by_day, TEMPERATURE

from .ends_fit import (
    EVENING,
    MORNING,
    LinRegModel,
    create_end_profile,
    create_morning_and_evening_models,
)
from .peak_fit.helpers import get_max_temp_and_time
from .peak_fit.optimize import (
    create_r_peak_temp_model,
    fit_peak_profile,
    prepare_many_fits,
)

# TODO => place where insert larger df for training needs to be clear!

Models = NamedTuple("Models", [("Peak", pwlf.PiecewiseLinFit), ("Morning", LinRegModel), ("Evening", LinRegModel)])




def init_models():
    fits_df = prepare_many_fits(n_days=28)  # TODO should be able to input df also
    r_peak_temp_model = create_r_peak_temp_model(fits_df)

    end_models = create_morning_and_evening_models()
    return Models(r_peak_temp_model, *end_models)


def fit_day_profile(
    df: pl.DataFrame,
    day: int,
    models: Models
):
    morn_profile = create_end_profile(df, day, MORNING, models.Morning)
    eve_profile = create_end_profile(df, day, EVENING, models.Evening)
    peak_profile = fit_peak_profile(df, day, models.Peak)


    def create_df(profile:Profile, name:str):
        day_diff = day - profile.Hours[0].day
        updated_times = [i.add(days=day_diff) for i in profile.Hours]
        return pl.DataFrame({
            "datetime": updated_times,
            "values": profile.Values,
            "names": [name]*len(profile.Hours)
        })

    return pl.concat([create_df(profile, name) for profile, name in zip([morn_profile, eve_profile, peak_profile], ["morn", "eve", "peak"])])


def plot_day_profile(profiles_df: pl.DataFrame, df:pl.DataFrame, day:int):
    base = alt.Chart(filter_by_day(df, day)).mark_line().encode(
        alt.X("datetime"),
        alt.Y(TEMPERATURE),
        color=alt.value("grey")
    )
    fit =  alt.Chart(profiles_df).mark_line(point=True).encode(
        alt.X("datetime"),
        alt.Y("values").scale(zero=False),
        alt.Color("names")
    )

    return base + fit