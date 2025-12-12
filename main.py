
"""Main entry point for Part 1 of the trading bot project.

This script wires together:
- data loading
- feature engineering
- labeling
- time-based splitting
- model training
- evaluation

Usage:
    python main.py

Make sure to adjust config.py (especially data_path, SL/TP, and horizon) to match
your instrument (e.g., XAUUSD M15, NAS100 H1, EURUSD M5, etc.).
"""

from config import Config
from src.data_loader import load_data
from src.features import engineer_features
from src.labeling import add_labels
from src.splitter import time_split, get_feature_target_matrices
from src.modeling import train_model, evaluate_on_test
from src.visualization import plot_label_window
from src.visualization_trades import plot_test_trades
from src.equity_curve import build_equity_curve, plot_equity_curve



def main():
    cfg = Config()

    # 1) Load raw data
    df = load_data(cfg)
    print(f"Loaded {len(df)} rows of data.")

    # 2) Engineer features
    df_feat = engineer_features(df)
    print(f"After feature engineering and dropping NaNs: {len(df_feat)} rows remain.")

    # 3) Add labels based on SL/TP and horizon
    df_lab = add_labels(df_feat, cfg.sl, cfg.tp, cfg.horizon)
    print("Label distribution:")
    print(df_lab["label"].value_counts())

    # If you want to drop no-trade labels (0) and train only on BUY/SELL:
    # df_lab = df_lab[df_lab["label"] != 0].reset_index(drop=True)

    # Visualize the most recent N candles with labels
    # plot_label_window(
    #     df_lab,
    #     n=40000,  # last 300 candles; you can change this
    #     start_idx=None,
    #     title="Recent price action with BUY/SELL labels",
    # )

    # 4) Time-based split
    df_train, df_val, df_test = time_split(df_lab, cfg.train_frac, cfg.val_frac)
    print(
        f"Split sizes: train={len(df_train)}, val={len(df_val)}, test={len(df_test)}"
    )

    # 5) Build feature matrices
    X_train, y_train = get_feature_target_matrices(df_train)
    X_val, y_val = get_feature_target_matrices(df_val)
    X_test, y_test = get_feature_target_matrices(df_test)

    # 6) Train the model
    clf = train_model(X_train, y_train, X_val, y_val, cfg)

    # 7) Evaluate on test
    y_pred = evaluate_on_test(clf, X_test, y_test, cfg)

    plot_test_trades(
        df_test=df_test,
        y_test=y_test,
        y_pred=y_pred,
        n=8000,
        start_idx=None,
        title="TEST set trades (raw predictions): wins vs losses",
    )

    equity_df = build_equity_curve(
        df_test=df_test,
        y_test=y_test,
        y_pred=y_pred,
        start_capital=10_000,
        risk_mode="percent",
        risk_value=0.01,      # 1% risk per trade
        R=cfg.tp / cfg.sl,    # or just 2.0
        cost_per_trade_R=0.02, # optional (e.g., 0.02R per trade)
        ignore_true_zero=True
    )

    plot_equity_curve(equity_df, title="Test Set Equity Curve (prob-filtered trades)")





if __name__ == "__main__":
    main()
