# src/equity_curve.py

from typing import Optional, Dict, Any
import pandas as pd
import matplotlib.pyplot as plt


def build_equity_curve(
    df_test: pd.DataFrame,
    y_test,
    y_pred,
    start_capital: float = 10_000.0,
    risk_mode: str = "percent",   # "percent" or "fixed"
    risk_value: float = 0.01,     # 1% if percent, or $ amount if fixed
    R: float = 2.0,
    cost_per_trade_R: float = 0.0,  # costs in R units (e.g., 0.05R)
    ignore_true_zero: bool = True,  # if True, skip trades where y_true == 0
) -> pd.DataFrame:
    """
    Builds an equity curve based on model trades on the test set.

    Parameters
    ----------
    df_test : DataFrame with at least ['time'] (and preferably 'close')
    y_test : true labels (-1,0,1)
    y_pred : model actions (-1,0,1) AFTER your probability filter
    start_capital : initial account equity
    risk_mode : "percent" or "fixed"
    risk_value : if percent -> fraction of equity risked per trade (e.g., 0.01)
                 if fixed -> fixed dollar amount risked per trade (e.g., 100)
    R : reward-to-risk ratio (TP/SL)
    cost_per_trade_R : transaction costs per trade expressed in R units
    ignore_true_zero : if True, skip trades whose y_true == 0 (ambiguous outcomes)

    Returns
    -------
    DataFrame with equity over time for taken trades.
    """
    df = df_test.copy()
    df["time"] = pd.to_datetime(df["time"])
    df["y_true"] = list(y_test)
    df["y_pred"] = list(y_pred)

    # Taken trades are y_pred != 0
    trades = df[df["y_pred"] != 0].copy()

    if ignore_true_zero:
        trades = trades[trades["y_true"] != 0].copy()

    equity = start_capital
    rows = []

    for _, row in trades.iterrows():
        # Determine risk amount
        if risk_mode == "percent":
            risk_amt = equity * float(risk_value)
        elif risk_mode == "fixed":
            risk_amt = float(risk_value)
        else:
            raise ValueError("risk_mode must be 'percent' or 'fixed'")

        # Outcome
        win = (row["y_true"] == row["y_pred"]) and (row["y_true"] != 0)

        # Profit/Loss in $ (apply transaction cost in R units)
        if win:
            pnl = risk_amt * (R - cost_per_trade_R)
            outcome = "win"
        else:
            pnl = -risk_amt * (1 + cost_per_trade_R)
            outcome = "loss"

        equity += pnl

        rows.append({
            "time": row["time"],
            "y_pred": int(row["y_pred"]),
            "y_true": int(row["y_true"]),
            "outcome": outcome,
            "risk_amt": risk_amt,
            "pnl": pnl,
            "equity": equity,
        })

    return pd.DataFrame(rows)


def plot_equity_curve(equity_df: pd.DataFrame, title: str = "Equity Curve (Test Set)") -> None:
    if equity_df.empty:
        print("Equity DF is empty (no trades). Nothing to plot.")
        return

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(equity_df["time"], equity_df["equity"], linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Equity")
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show()

    # Quick summary
    start_eq = equity_df["equity"].iloc[0] - equity_df["pnl"].iloc[0]
    end_eq = equity_df["equity"].iloc[-1]
    total_pnl = equity_df["pnl"].sum()
    wins = (equity_df["outcome"] == "win").sum()
    losses = (equity_df["outcome"] == "loss").sum()

    print("\n=== Equity Summary ===")
    print(f"Trades: {len(equity_df)} | Wins: {wins} | Losses: {losses}")
    print(f"Start equity: {start_eq:.2f}")
    print(f"End equity:   {end_eq:.2f}")
    print(f"Total PnL:    {total_pnl:.2f}")
