import pendulum
from numpy import ndarray
from typing import NamedTuple

Profile = NamedTuple("Profile", [("Values", ndarray), ("Hours", list[pendulum.DateTime | pendulum.Date])])