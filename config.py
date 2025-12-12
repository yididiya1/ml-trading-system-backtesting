
from dataclasses import dataclass

@dataclass
class Config:
    """Global configuration for Part 1 ML pipeline."""
    # Path to your historical data CSV
    data_path: str = "Data/XAUUSD_M5.csv"


    # Minimum probability required to take a BUY/SELL trade
    min_prob_trade: float = 0.45 

    # Column names in your CSV
    time_col: str = "time"   # or "timestamp"
    open_col: str = "open"
    high_col: str = "high"
    low_col: str = "low"
    close_col: str = "close"
    volume_col: str = "volume"

    # Labeling parameters (in absolute price units)
    sl: float = 5.0       # stop loss distance (e.g. 2.0 = $2 for gold)
    tp: float = 15.0       # take profit distance
    horizon: int = 300    # how many future candles we look ahead

    # Train/val/test split (by time)
    train_frac: float = 0.7
    val_frac: float = 0.15
    test_frac: float = 0.15  # must sum to 1 with others

    # Random Forest hyperparameters
    n_estimators: int = 200
    max_depth: int = 10
    random_state: int = 42
