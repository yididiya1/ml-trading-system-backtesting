
import numpy as np
import pandas as pd


def add_labels(df: pd.DataFrame, sl: float, tp: float, horizon: int) -> pd.DataFrame:
    """Create 3-class labels based on SL/TP and forward horizon.

    For each candle t:
    - Enter at close[t]
    - BUY scenario: TP at close[t] + tp, SL at close[t] - sl
    - SELL scenario: TP at close[t] - tp, SL at close[t] + sl
    - Look forward up to `horizon` candles.
    - Determine whether BUY TP or BUY SL is hit first, and similarly for SELL.
    - Label:
        +1 if BUY wins (TP hit first) and SELL does not also clearly win first
        -1 if SELL wins and BUY does not also win first
         0 otherwise (no clear edge / ambiguous / both lose)
    """
    df = df.copy()
    n = len(df)
    labels = np.zeros(n, dtype=int)  # default 0 = no-trade / ambiguous

    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values

    for i in range(n):
        entry = closes[i]

        # BUY levels
        tp_buy = entry + tp
        sl_buy = entry - sl

        # SELL levels
        tp_sell = entry - tp
        sl_sell = entry + sl

        buy_outcome = None   # 1 = TP hit first, -1 = SL hit first, None = unresolved
        sell_outcome = None

        # Look ahead up to 'horizon' candles, or until end of data
        for j in range(i + 1, min(i + 1 + horizon, n)):
            hi = highs[j]
            lo = lows[j]

            # ----- BUY scenario -----
            if buy_outcome is None:
                hit_tp_buy = hi >= tp_buy
                hit_sl_buy = lo <= sl_buy

                if hit_tp_buy and not hit_sl_buy:
                    buy_outcome = 1   # BUY TP first
                elif hit_sl_buy and not hit_tp_buy:
                    buy_outcome = -1  # BUY SL first
                elif hit_tp_buy and hit_sl_buy:
                    # both in same candle (whipsaw) -> assume worst case for us
                    buy_outcome = -1

            # ----- SELL scenario -----
            if sell_outcome is None:
                hit_tp_sell = lo <= tp_sell
                hit_sl_sell = hi >= sl_sell

                if hit_tp_sell and not hit_sl_sell:
                    sell_outcome = 1   # SELL TP first
                elif hit_sl_sell and not hit_tp_sell:
                    sell_outcome = -1  # SELL SL first
                elif hit_tp_sell and hit_sl_sell:
                    # both in same candle -> assume worst case
                    sell_outcome = -1

            # If both resolved, we can stop scanning forward
            if (buy_outcome is not None) and (sell_outcome is not None):
                break

        # ----- Decide the final label for candle i -----
        if buy_outcome == 1 and sell_outcome != 1:
            labels[i] = 1   # BUY label
        elif sell_outcome == 1 and buy_outcome != 1:
            labels[i] = -1  # SELL label
        else:
            labels[i] = 0   # NO-TRADE / ambiguous

    df["label"] = labels
    return df
