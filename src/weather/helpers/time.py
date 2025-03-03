import pendulum
from typing import NamedTuple

from weather.fit.peak_fit.helpers import EXTENT

Day = NamedTuple("Day", [("year", int),("month", int),("day", int)])

def get_month(df):
    adate = pendulum.instance(df["datetime"][0])
    return adate.format("MMM")



def create_centered_interval(
    peak_hour: int, extent=EXTENT, day=Day(2020, 6, 1)
):
    time = pendulum.naive(*day, hour=peak_hour)

    start = time.subtract(hours=extent)
    end = time.add(hours=extent)
    interval = pendulum.interval(start, end)
    return [dt for dt in interval.range("hours")]


def create_interval(start_hour:int, end_hour:int, day=Day(2020, 6, 1) ):
    start = pendulum.naive(*day, hour=start_hour)
    end = pendulum.naive(*day, hour=end_hour)
    interval = pendulum.interval(start, end)
    return [dt for dt in interval.range("hours")]