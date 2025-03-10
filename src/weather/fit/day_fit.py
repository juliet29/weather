from typing import NamedTuple

import altair as alt
import polars as pl
from scipy.stats import kstest
import pwlf

from ..helpers.dataframes import TEMPERATURE, filter_by_day
from ..helpers.time import HOURS_IN_DAY
from .ends_fit import (
    EVENING,
    MORNING,
    LinRegModel,
    create_end_profile,
    create_morning_and_evening_models,
)
from .interfaces import Profile
from .peak_fit.optimize import (
    create_r_peak_temp_model,
    fit_peak_profile,
    prepare_many_fits,
)

Models = NamedTuple(
    "Models",
    [
        ("Peak", pwlf.PiecewiseLinFit),
        ("Morning", LinRegModel),
        ("Evening", LinRegModel),
    ],
)


def init_models():
    fits_df = prepare_many_fits(n_days=28)  # TODO should be able to input df also
    r_peak_temp_model = create_r_peak_temp_model(fits_df)

    end_models = create_morning_and_evening_models()
    return Models(r_peak_temp_model, *end_models)


def fit_day_profile(df: pl.DataFrame, days: list[int], models: Models):
    def create_profile_df(day: int, profile: Profile, name: str):
        day_diff = day - profile.Hours[0].day
        updated_times = [i.add(days=day_diff) for i in profile.Hours]
        return pl.DataFrame(
            {
                "datetime": updated_times,
                "values": profile.Values,
                "names": [name] * len(profile.Hours),
                "day": [day] * len(profile.Hours),
            }
        )

    def create_day_df(day):
        morn_profile = create_end_profile(df, day, MORNING, models.Morning)
        eve_profile = create_end_profile(df, day, EVENING, models.Evening)
        peak_profile = fit_peak_profile(df, day, models.Peak)

        profile_df = pl.concat(
            [
                create_profile_df(day, profile, name)
                for profile, name in zip(
                    [morn_profile, eve_profile, peak_profile], ["morn", "eve", "peak"]
                )
            ]
        )
        day_df = filter_by_day(df, day).join(profile_df, on=["datetime"]).rename({TEMPERATURE: "true_temp", "values": "fit_temp"})
        assert day_df["datetime"].n_unique() == HOURS_IN_DAY
        return day_df

    return pl.concat([create_day_df(d) for d in days])


def test_similarity(profiles_df:pl.DataFrame):
    res = kstest(profiles_df["true_temp"], profiles_df["fit_temp"])
    print(res)

    P_THRESHOLD = 0.05
    p = res.pvalue

    print("\n----------------------")
    print("Null hypothesis: samples are drawn from the same distribution.")
    if p < P_THRESHOLD:
        print(f"P-value {p:.3f} < {P_THRESHOLD}. Reject the null hypothesis - samples ARE NOT  drawn from the same distribution.")
    else:
        print(f"P-value {p:.3f} >= {P_THRESHOLD}. Cannot reject the null hypothesis - samples MAY be drawn from the same distribution.")

    return res


def plot_day_profile(profiles_df: pl.DataFrame):
    base = (
        alt.Chart(profiles_df)
        .mark_line()
        .encode(alt.X("hours(datetime)"), alt.Y("true_temp"), color=alt.value("grey"))
        .properties(width=100, height=100)
    )
    fit = base.mark_line(point=True).encode(
        alt.X("hours(datetime)"), alt.Y("fit_temp").scale(zero=False), alt.Color("names")
    )

    return (base + fit).facet("day", columns=4)


def plot_distributions(profiles_df: pl.DataFrame):
    dist_df = profiles_df.unpivot(
        on=["true_temp", "fit_temp"], index="datetime"
    )

    return (
        alt.Chart(dist_df)
        .transform_density(
            density="value",
            bandwidth=0.5,
            counts=True,
            steps=200,
            groupby=["variable"],
        )
        .mark_area(opacity=0.5)
        .encode(
            alt.X("value:Q"), alt.Y("density:Q").stack(None), alt.Color("variable:N")
        )
    )


def plot_histogram(profiles_df: pl.DataFrame):
    dist_df = profiles_df.unpivot(
        on=["true_temp", "fit_temp"], index="datetime"
    )

    return alt.Chart(dist_df).mark_bar(
        opacity=0.5
    ).encode(
        alt.X("value:Q").bin(maxbins=30),
        alt.Y("count()").stack(None),
        alt.Color("variable:N")
    )