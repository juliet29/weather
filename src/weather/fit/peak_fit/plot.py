import altair as alt
import polars as pl


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

    res = (base + fit + text1).facet(facet="day", columns=4)

    return res
