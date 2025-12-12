# src/visualization.py

import matplotlib.pyplot as plt
import pandas as pd
from typing import Optional


def plot_label_window(
    df: pd.DataFrame,
    n: int = 300,
    start_idx: Optional[int] = None,
    title: str = "Price with BUY/SELL labels",
) -> None:
    """
    Plot a window of data with BUY / SELL labels overlaid on the close price.

    Parameters
    ----------
    df : DataFrame
        Must contain columns: time, close, label
    n : int
        Number of candles to plot (if start_idx is None, uses the last n)
    start_idx : int or None
        Index to start plotting from. If None, uses the last n rows.
    title : str
        Plot title
    """
    if "time" not in df.columns or "close" not in df.columns or "label" not in df.columns:
        raise ValueError("DataFrame must contain 'time', 'close', and 'label' columns")

    if start_idx is None:
        # Take the last n rows
        window = df.iloc[-n:].copy()
    else:
        window = df.iloc[start_idx : start_idx + n].copy()

    if window.empty:
        print("Window is empty, nothing to plot.")
        return

    # Make sure time is datetime
    window["time"] = pd.to_datetime(window["time"])

    # Split by label
    buys = window[window["label"] == 1]
    sells = window[window["label"] == -1]
    neutrals = window[window["label"] == 0]

    fig, ax = plt.subplots(figsize=(12, 5))

    # Plot close price
    ax.plot(window["time"], window["close"], linewidth=1, label="Close")

    # Plot BUY / SELL markers
    if not buys.empty:
        ax.scatter(
            buys["time"],
            buys["close"],
            marker="^",
            s=40,
            label="BUY label",
        )

    if not sells.empty:
        ax.scatter(
            sells["time"],
            sells["close"],
            marker="v",
            s=40,
            label="SELL label",
        )

    # (Optional) show where 0-labels are, commented out for now:
    # if not neutrals.empty:
    #     ax.scatter(
    #         neutrals["time"],
    #         neutrals["close"],
    #         marker=".",
    #         s=10,
    #         alpha=0.4,
    #         label="NO-TRADE label",
    #     )

    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    ax.legend()
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show()
