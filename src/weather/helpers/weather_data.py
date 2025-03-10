from pathlib import Path
from typing import Literal, NamedTuple
from ..config import WEATHER_DATA


class WeatherData(NamedTuple):
    path: Path
    start_year: int
    dtype: Literal["EPW", "CSV"] 
    end_year: int | None = None 


PALO_ALTO_20 = WeatherData(WEATHER_DATA / "CA_PALO-ALTO-AP_724937S_20.epw", start_year=2020, dtype="EPW")

STANFORD_10 = WeatherData(WEATHER_DATA / "Stanford_10_24.csv", start_year=2010, end_year=2024, dtype="CSV")

