from pathlib import Path
import csv
import polars as pl
import pandas as pd


class epw:
    """A class which represents an EnergyPlus weather (epw) file"""

    def __init__(self):
        """ """
        self.headers = {}
        self.dataframe = pd.DataFrame()

    def read(self, fp):
        """Reads an epw file

        Arguments:
            - fp (str): the file path of the epw file

        """

        self.headers = self._read_headers(fp)
        self.dataframe = self._read_data(fp)

    def _read_headers(self, fp):
        """Reads the headers of an epw file

        Arguments:
            - fp (str): the file path of the epw file

        Return value:
            - d (dict): a dictionary containing the header rows

        """

        d = {}
        with open(fp, newline="") as csvfile:
            csvreader = csv.reader(csvfile, delimiter=",", quotechar='"')
            for row in csvreader:
                if row[0].isdigit():
                    break
                else:
                    d[row[0]] = row[1:]
        return d

    def _read_data(self, fp):
        """Reads the climate data of an epw file

        Arguments:
            - fp (str): the file path of the epw file

        Return value:
            - df (pd.DataFrame): a DataFrame comtaining the climate data

        """

        names = [
            "Year",
            "Month",
            "Day",
            "Hour",
            "Minute",
            "Data Source and Uncertainty Flags",
            "Dry Bulb Temperature",
            "Dew Point Temperature",
            "Relative Humidity",
            "Atmospheric Station Pressure",
            "Extraterrestrial Horizontal Radiation",
            "Extraterrestrial Direct Normal Radiation",
            "Horizontal Infrared Radiation Intensity",
            "Global Horizontal Radiation",
            "Direct Normal Radiation",
            "Diffuse Horizontal Radiation",
            "Global Horizontal Illuminance",
            "Direct Normal Illuminance",
            "Diffuse Horizontal Illuminance",
            "Zenith Luminance",
            "Wind Direction",
            "Wind Speed",
            "Total Sky Cover",
            "Opaque Sky Cover (used if Horizontal IR Intensity missing)",
            "Visibility",
            "Ceiling Height",
            "Present Weather Observation",
            "Present Weather Codes",
            "Precipitable Water",
            "Aerosol Optical Depth",
            "Snow Depth",
            "Days Since Last Snowfall",
            "Albedo",
            "Liquid Precipitation Depth",
            "Liquid Precipitation Quantity",
        ]

        first_row = self._first_row_with_climate_data(fp)
        df = pl.read_csv(fp, skip_rows=first_row, has_header=False, new_columns=names)
        return df

    def _first_row_with_climate_data(self, fp):
        """Finds the first row with the climate data of an epw file

        Arguments:
            - fp (str): the file path of the epw file

        Return value:
            - i (int): the row number

        """

        with open(fp, newline="") as csvfile:
            csvreader = csv.reader(csvfile, delimiter=",", quotechar='"')
            for i, row in enumerate(csvreader):
                if row[0].isdigit():
                    break
        return i


def clean_up_df(df: pl.DataFrame):
    return (
        (
            df.with_columns(pl.col("Hour") - 1)
            .insert_column(
                0,
                pl.concat_str(
                    [
                        pl.col("Year"),
                        pl.col("Month").cast(pl.String).str.pad_start(2, "0"),
                        pl.col("Day").cast(pl.String).str.pad_start(2, "0"),
                        pl.col("Hour").cast(pl.String).str.pad_start(2, "0"),
                        pl.col("Minute").cast(pl.String).str.pad_start(2, "0"),
                    ],
                    separator="-",
                ).alias("datetime"),
            )
            .with_columns(pl.col("datetime").str.to_datetime(format="%Y-%m-%d-%H-%M"))
        )
        .drop(["Year", "Month", "Day", "Hour", "Minute"])
        .drop(
            [
                "Data Source and Uncertainty Flags",
                "Present Weather Codes",
                "Atmospheric Station Pressure",
            ]
        )
    )


def read_epw(file: Path):
    assert file.exists
    e = epw()
    e.read(file)
    return clean_up_df(e.dataframe)
