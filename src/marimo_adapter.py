"""Model backend selection.

`get_estimator(cfg)` returns an estimator compatible with the project's
`train_model` / `evaluate_on_test` flow.
"""
from typing import Any

try:
    import marimo  # type: ignore
    MARIMO_AVAILABLE = True
except Exception:
    marimo = None  # type: ignore
    MARIMO_AVAILABLE = False

from sklearn.ensemble import RandomForestClassifier


def get_estimator(cfg: Any) -> Any:
    """Return an estimator instance configured from `cfg`."""
    if MARIMO_AVAILABLE:
        try:
            # Try a hypothetical marimo.models.RandomForest
            RF = getattr(getattr(marimo, "models", marimo), "RandomForest", None)
            if RF is not None:
                return RF(n_estimators=cfg.n_estimators, max_depth=cfg.max_depth, random_state=cfg.random_state)

            # Try a direct constructor
            RF2 = getattr(marimo, "RandomForest", None)
            if RF2 is not None:
                return RF2(n_estimators=cfg.n_estimators, max_depth=cfg.max_depth, random_state=cfg.random_state)

        except Exception:
            pass

    # Default: sklearn RandomForest.
    return RandomForestClassifier(
        n_estimators=getattr(cfg, "n_estimators", 100),
        max_depth=getattr(cfg, "max_depth", None),
        random_state=getattr(cfg, "random_state", 42),
        n_jobs=-1,
        class_weight="balanced",
    )
