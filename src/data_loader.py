
import pandas as pd
from . import typing_aliases as ta  # noqa: F401  # for potential future type aliases
from config import Config


def load_data(cfg: Config) -> pd.DataFrame:
    """
    CSV layout:
    time,open,high,low,close,volume
    2023-03-08 04:10:00,1810.595,1810.605,1809.958,1810.138,5,307

    Here "5" is timeframe (we drop it), "307" is actual volume.
    """

    df = pd.read_csv(cfg.data_path)

    print("Orginal Data")
    print(df.head())


    df.columns = ["time", "open", "high", "low", "close", "tf", "volume"]
    

    # Drop the timeframe column if present
    if "tf" in df.columns:
        df = df[["time", "open", "high", "low", "close", "volume"]]

    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)
    return df

