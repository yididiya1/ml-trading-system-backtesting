
from typing import Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

from config import Config


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    cfg: Config,
) -> RandomForestClassifier:
    """Train a RandomForestClassifier and print validation metrics."""
    clf = RandomForestClassifier(
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        random_state=cfg.random_state,
        n_jobs=-1,
        class_weight="balanced",  # helps if labels are imbalanced
    )
    clf.fit(X_train, y_train)

    # Validation metrics
    y_val_pred = clf.predict(X_val)
    print("=== Validation Classification Report ===")
    print(classification_report(y_val, y_val_pred, digits=3))
    print("=== Validation Confusion Matrix ===")
    print(confusion_matrix(y_val, y_val_pred))

    return clf


# def evaluate_on_test(
#     clf: RandomForestClassifier,
#     X_test: pd.DataFrame,
#     y_test: pd.Series,
#     cfg: Config,
# ) -> None:
#     """Evaluate on test set and estimate expectancy per trade based on SL/TP ratio."""
#     y_pred = clf.predict(X_test)

#     print("\n=== TEST SET PERFORMANCE ===")
#     print(classification_report(y_test, y_pred, digits=3))
#     print("=== Test Confusion Matrix ===")
#     print(confusion_matrix(y_test, y_pred))

#     # Simple trade expectancy estimation:
#     # Assume:
#     #  - Correct directional trades yield +R (R = tp/sl)
#     #  - Wrong directional trades yield -1 (1R loss)
#     #  - NO-TRADE (0) yields 0
#     R = cfg.tp / cfg.sl if cfg.sl != 0 else 1.0

#     # Only consider candles where we actually traded (pred != 0)
#     trade_mask = y_pred != 0
#     y_test_trades = y_test[trade_mask]
#     y_pred_trades = y_pred[trade_mask]

#     if len(y_test_trades) == 0:
#         print("\nNo trades taken on test set based on predictions.")
#         return

#     # A "win" is when direction matches sign and label != 0
#     wins = ((y_test_trades == y_pred_trades) & (y_test_trades != 0)).sum()
#     losses = (y_test_trades != 0).sum() - wins

#     total_trades = wins + losses
#     win_rate = wins / total_trades if total_trades > 0 else 0.0

#     expectancy_per_trade = (win_rate * R) + ((1 - win_rate) * -1)

#     print(f"\nTrades on test set: {total_trades}")
#     print(f"Win rate (directional): {win_rate:.3f}")
#     print(f"R (TP/SL): {R:.2f}")
#     print(f"Estimated expectancy per trade (in R): {expectancy_per_trade:.3f}")





def evaluate_on_test(
    clf: RandomForestClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cfg: Config,
) -> None:
    """
    Evaluate on test set with a probability filter:
    - Only take trades when max class probability for BUY/SELL exceeds cfg.min_prob_trade.
    - Otherwise treat it as NO-TRADE.
    """

    # Raw class predictions (for reference)
    y_pred_raw = clf.predict(X_test)
    print("\n=== TEST SET PERFORMANCE (raw predictions, no prob filter) ===")
    print(classification_report(y_test, y_pred_raw, digits=3))
    print("=== Test Confusion Matrix (raw) ===")
    print(confusion_matrix(y_test, y_pred_raw))

    # ---- Probability-based filtered predictions ----
    # Get class probabilities: shape (n_samples, n_classes)
    proba = clf.predict_proba(X_test)  # columns correspond to classes in clf.classes_

    classes = clf.classes_
    # Find index of labels -1, 0, 1 in the proba array
    idx_minus1 = np.where(classes == -1)[0][0]
    idx_zero = np.where(classes == 0)[0][0]
    idx_plus1 = np.where(classes == 1)[0][0]

    # For each sample, decide:
    # - compute p_sell, p_no_trade, p_buy
    # - take the best non-zero class only if its prob > min_prob_trade
    # - else, prediction = 0
    filtered_preds = []
    for i in range(len(X_test)):
        p_sell = proba[i, idx_minus1]
        p_zero = proba[i, idx_zero]
        p_buy = proba[i, idx_plus1]

        # best non-zero class & its prob
        if p_buy >= p_sell:
            best_label = 1
            best_prob = p_buy
        else:
            best_label = -1
            best_prob = p_sell

        if best_prob >= cfg.min_prob_trade:
            filtered_preds.append(best_label)
        else:
            filtered_preds.append(0)  # skip trade

    y_pred = np.array(filtered_preds)

    print(f"\n=== TEST SET PERFORMANCE with prob filter (min_prob_trade={cfg.min_prob_trade}) ===")
    print(classification_report(y_test, y_pred, digits=3))
    print("=== Test Confusion Matrix (prob-filtered) ===")
    print(confusion_matrix(y_test, y_pred))

    # ---- Trading expectancy with prob-filtered trades ----
    R = cfg.tp / cfg.sl if cfg.sl != 0 else 1.0

    trade_mask = y_pred != 0
    y_test_trades = y_test[trade_mask]
    y_pred_trades = y_pred[trade_mask]

    if len(y_test_trades) == 0:
        print("\nNo trades taken on test set based on probability filter.")
        return

    wins = ((y_test_trades == y_pred_trades) & (y_test_trades != 0)).sum()
    losses = (y_test_trades != 0).sum() - wins

    total_trades = wins + losses
    win_rate = wins / total_trades if total_trades > 0 else 0.0
    expectancy_per_trade = (win_rate * R) + ((1 - win_rate) * -1)

    print(f"\nTrades on test set (prob-filtered): {total_trades}")
    print(f"Win rate (directional, prob-filtered): {win_rate:.3f}")
    print(f"R (TP/SL): {R:.2f}")
    print(f"Estimated expectancy per trade (in R, prob-filtered): {expectancy_per_trade:.3f}")


    return y_pred
