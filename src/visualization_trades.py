# src/visualization_trades.py

import matplotlib.pyplot as plt
import pandas as pd
from typing import Optional


def plot_test_trades(
    df_test: pd.DataFrame,
    y_test,
    y_pred,
    n: int = 600,
    start_idx: Optional[int] = None,
    title: str = "Test set trades: wins vs losses",
) -> None:
    """
    Visualize model trades on the TEST set.
    - Entry markers are placed where y_pred != 0 (trade taken).
    - Marker color indicates outcome vs y_test:
        green = win (pred == true and true != 0)
        red   = loss (pred != true, true != 0)
        orange = predicted trade but y_test == 0 (should not happen often; means labeler says no clear outcome)
    - Marker shape indicates direction:
        '^' for BUY, 'v' for SELL

    df_test must contain columns: 'time', 'close'
    """
    if "time" not in df_test.columns or "close" not in df_test.columns:
        raise ValueError("df_test must contain 'time' and 'close' columns")

    df = df_test.copy()
    df["time"] = pd.to_datetime(df["time"])

    # Attach arrays
    df["y_true"] = list(y_test)
    df["y_pred"] = list(y_pred)

    # Window selection
    if start_idx is None:
        window = df.iloc[-n:].copy()
    else:
        window = df.iloc[start_idx : start_idx + n].copy()

    if window.empty:
        print("Window is empty, nothing to plot.")
        return

    # Trades are where prediction is BUY/SELL
    trades = window[window["y_pred"] != 0].copy()

    # Determine outcome category
    def outcome(row):
        if row["y_true"] == 0:
            return "ambiguous"  # labeler says no TP/SL hit within horizon
        return "win" if row["y_true"] == row["y_pred"] else "loss"

    trades["outcome"] = trades.apply(outcome, axis=1)

    # Split for plotting
    buy_wins = trades[(trades["y_pred"] == 1) & (trades["outcome"] == "win")]
    buy_losses = trades[(trades["y_pred"] == 1) & (trades["outcome"] == "loss")]
    sell_wins = trades[(trades["y_pred"] == -1) & (trades["outcome"] == "win")]
    sell_losses = trades[(trades["y_pred"] == -1) & (trades["outcome"] == "loss")]
    ambiguous = trades[trades["outcome"] == "ambiguous"]

    # # Plot close
    # fig, ax = plt.subplots(figsize=(14, 6))
    # ax.plot(window["time"], window["close"], linewidth=1, label="Close")

    # # Plot entries: colors for win/loss, shapes for direction
    # if not buy_wins.empty:
    #     ax.scatter(buy_wins["time"], buy_wins["close"], marker="^", s=60, label="BUY win")
    # if not buy_losses.empty:
    #     ax.scatter(buy_losses["time"], buy_losses["close"], marker="^", s=60, label="BUY loss")
    # if not sell_wins.empty:
    #     ax.scatter(sell_wins["time"], sell_wins["close"], marker="v", s=60, label="SELL win")
    # if not sell_losses.empty:
    #     ax.scatter(sell_losses["time"], sell_losses["close"], marker="v", s=60, label="SELL loss")
    # if not ambiguous.empty:
    #     ax.scatter(ambiguous["time"], ambiguous["close"], marker="o", s=40, label="Trade but true label=0")

    # ax.set_title(title)
    # ax.set_xlabel("Time")
    # ax.set_ylabel("Price")
    # ax.legend()
    # fig.autofmt_xdate()
    # plt.tight_layout()
    # plt.show()
    # --- Plot close price ---
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(window["time"], window["close"], linewidth=1, color="black", label="Close")

    # --- Wins ---
    ax.scatter(
        buy_wins["time"], buy_wins["close"],
        marker="^", color="green", s=70, label="BUY win"
    )
    ax.scatter(
        sell_wins["time"], sell_wins["close"],
        marker="v", color="green", s=70, label="SELL win"
    )

    # --- Losses ---
    ax.scatter(
        buy_losses["time"], buy_losses["close"],
        marker="^", color="red", s=70, label="BUY loss"
    )
    ax.scatter(
        sell_losses["time"], sell_losses["close"],
        marker="v", color="red", s=70, label="SELL loss"
    )

    # --- Ambiguous trades (true label = 0) ---
    buy_ambiguous = ambiguous[ambiguous["y_pred"] == 1]
    sell_ambiguous = ambiguous[ambiguous["y_pred"] == -1]

    ax.scatter(
        buy_ambiguous["time"], buy_ambiguous["close"],
        marker="^", color="blue", s=60, label="BUY (true label = 0)"
    )
    ax.scatter(
        sell_ambiguous["time"], sell_ambiguous["close"],
        marker="v", color="purple", s=60, label="SELL (true label = 0)"
    )

    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    ax.legend()
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show()


    print("\nTrade summary in plotted window:")
    print(trades["outcome"].value_counts(dropna=False))
    print(f"Total trades shown: {len(trades)}")
