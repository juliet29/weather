import pendulum
from ..config import FIGURES


def get_todays_save_path():
    today = pendulum.today().format("YYMMDD")
    path = FIGURES / today
    if not path.exists():
        path.mkdir()
    return path
