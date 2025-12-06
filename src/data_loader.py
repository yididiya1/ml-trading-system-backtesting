
import os
import pandas as pd
import numpy as np
from . import typing_aliases as ta  # noqa: F401  # for potential future type aliases
from config import Config


def load_data(cfg: Config) -> pd.DataFrame:
    """
    CSV layout:
    time,open,high,low,close,volume
    2023-03-08 04:10:00,1810.595,1810.605,1809.958,1810.138,5,307

    Here "5" is timeframe (we drop it), "307" is actual volume.
    """

    # If data file is missing, generate a small synthetic dataset so the
    # pipeline can run end-to-end for testing / development.
    if not os.path.exists(cfg.data_path):
        print(f"Data file {cfg.data_path} not found — generating synthetic sample.")
        periods = 500
        times = pd.date_range("2020-01-01", periods=periods, freq="5min")
        # simple random-walk prices
        np.random.seed(0)
        steps = np.random.normal(scale=0.1, size=periods)
        price = 1800 + np.cumsum(steps)
        high = price + np.abs(np.random.normal(0.05, 0.02, size=periods))
        low = price - np.abs(np.random.normal(0.05, 0.02, size=periods))
        open_ = np.concatenate([[price[0]], price[:-1]])
        close = price
        tf = [5] * periods
        volume = np.random.randint(100, 1000, size=periods)

        sample_df = pd.DataFrame({
            "time": times.astype(str),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "tf": tf,
            "volume": volume,
        })

        os.makedirs(os.path.dirname(cfg.data_path), exist_ok=True)
        sample_df.to_csv(cfg.data_path, index=False)

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

