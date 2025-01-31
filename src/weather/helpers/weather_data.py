from pathlib import Path
from typing import NamedTuple
from ..config import WEATHER_DATA


class WeatherData(NamedTuple):
    path: Path
    year: int


PALO_ALTO_20 = WeatherData(WEATHER_DATA / "CA_PALO-ALTO-AP_724937S_20.epw", year=2020)


