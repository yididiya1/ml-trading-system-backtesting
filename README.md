# ML Trading Bot – Part 1 (Offline Research & Backtesting)

This repository contains **Part 1** of a modular machine-learning trading system.  
The focus of Part 1 is **offline research**: loading historical OHLCV data, creating features, labeling trades using SL/TP + horizon, training a classifier, filtering trades by probability, and evaluating performance (including trade/equity visualizations).

> ⚠️ This project is **NOT** a live trading bot yet.  
> It does **not** connect to a broker or execute orders.  
> Part 1 is only for research/backtesting and signal validation.

---

## What this project does (high level)

Given historical OHLCV candles, the pipeline:

1. **Loads & cleans** historical OHLCV data
2. **Generates features** (returns, volatility, candle structure, rolling stats, etc.)
3. **Labels each candle** as:
   - `+1` BUY (TP hit before SL within horizon)
   - `-1` SELL (SL hit before TP within horizon)
   - `0` NO-TRADE (neither hit within horizon)
4. **Splits data by time** into train / validation / test
5. **Handles label imbalance** (optional undersampling of `0` labels in training)
6. **Trains a classifier** (RandomForest baseline)
7. **Applies probability filtering** to only take confident BUY/SELL predictions
8. **Evaluates performance**:
   - classification report
   - confusion matrix
   - trade count
   - directional win rate on taken trades
   - expectancy in R-multiples
9. **Visualizes**:
   - trades on test set (wins/losses/ambiguous)
   - optional equity curve (simulated)

---

## Project structure

Typical structure:

```bash
trading_bot_part1/
├─ main.py # Entry point: runs the entire pipeline
├─ requirements.txt # Python dependencies
├─ config.py # Strategy + pipeline configuration (SL/TP/horizon, thresholds, split sizes)
└─ src/
├─ data_loader.py # Load CSV and normalize columns/types
├─ features.py # Feature engineering
├─ labeling.py # Event-based labeling (SL/TP + horizon)
├─ split.py # Time-based split into train/val/test
├─ modeling.py # Train model (RandomForest baseline)
├─ evaluation.py # Performance reports + trade metrics
├─ visualization.py # Label visualization (optional)
├─ visualization_trades.py # Trade visualization on test set (win/loss/ambiguous)
└─ equity_curve.py # Simulated equity curve from test trades (optional)
```

## Requirements

### Python
- Python **3.9+** recommended

### Install dependencies
Create and activate a virtual environment (recommended):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Install packages
```bash
pip install -r requirements.txt
```

### Preparing historical data
Required CSV format

Your CSV should be clean and consistent with a header like:

```bash
time,open,high,low,close,volume
2023-03-08 04:10:00,1810.595,1810.605,1809.958,1810.138,307
...
```

Key requirements:
- time must be parseable as datetime
- columns must be numeric
- candles must be sorted in increasing time order

Notes
- If your raw export includes extra columns (like timeframe), either:
   - drop them before importing, OR
   - update data_loader.py to ignore/rename properly

### Configuration (SL/TP/Horizon, thresholds, splits)

Open config.py (or wherever your cfg object is defined). Typical parameters:

Labeling parameters
- sl : stop loss distance (in price units)
- tp : take profit distance (in price units)
- horizon : number of future candles to look ahead to decide label outcome

```
sl = 5
tp = 15
horizon = 300
```
Interpretation (entry at 1900):
- BUY: SL=1895, TP=1915
- SELL: SL=1905, TP=1885


### Probability filtering
- min_prob_trade : minimum confidence required to take BUY/SELL trade


### Data split settings
- train_frac, val_frac : time split ratios


