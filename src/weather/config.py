import pyprojroot


BASE_PATH = pyprojroot.find_root(pyprojroot.has_dir(".git"))

WEATHER_DATA =  BASE_PATH.parent / "weather_data"

FIGURES = BASE_PATH / "figures"


