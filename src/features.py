
import pandas as pd


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add simple technical features using ONLY past data (no future leakage).

    You can extend this later with more advanced indicators.
    """
    df = df.copy()

    # Basic returns
    df["return_1"] = df["close"].pct_change()         # 1-candle return
    df["return_3"] = df["close"].pct_change(3)
    df["return_5"] = df["close"].pct_change(5)

    # Moving averages
    df["sma_10"] = df["close"].rolling(10).mean()
    df["sma_20"] = df["close"].rolling(20).mean()
    df["ema_10"] = df["close"].ewm(span=10, adjust=False).mean()
    df["ema_20"] = df["close"].ewm(span=20, adjust=False).mean()

    # Simple volatility features
    df["volatility_10"] = df["return_1"].rolling(10).std()
    df["volatility_20"] = df["return_1"].rolling(20).std()

    # ATR-like feature (rough)
    high_low = df["high"] - df["low"]
    high_close_prev = (df["high"] - df["close"].shift(1)).abs()
    low_close_prev = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
    df["atr_14"] = tr.rolling(14).mean()

    # Relative position vs moving averages
    df["close_vs_sma_20"] = df["close"] / df["sma_20"] - 1.0
    df["close_vs_ema_20"] = df["close"] / df["ema_20"] - 1.0

    # Time-based features (if time is in UTC or consistent)
    df["hour"] = df["time"].dt.hour
    df["dayofweek"] = df["time"].dt.dayofweek

    # Drop initial NaNs from rolling indicators
    df = df.dropna().reset_index(drop=True)

    print("Engineered features")
    print(df.head())

    return df
