"""Pair classifier: scikit-learn HistGradientBoosting (threads capped by
OMP_NUM_THREADS via machine_limits). LightGBM would be the usual choice, but its
macOS wheel needs a system libomp that this machine lacks."""
import machine_limits  # noqa: F401  -- must stay the first import (caps CPU threads)
import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingClassifier

SEED = 42
# identifiers / labels / bookkeeping columns, never used as features
NON_FEATURES = {"s1_rid", "m_rid", "bits", "s1_id", "match_id", "label", "country"}


def feature_columns(df):
    return [c for c in df.columns if c not in NON_FEATURES]


def to_matrix(df, cols):
    return df.select(pl.col(cols).cast(pl.Float32)).to_numpy()


def make_model():
    return HistGradientBoostingClassifier(
        max_iter=1500,
        learning_rate=0.1,
        max_leaf_nodes=63,
        min_samples_leaf=50,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=30,
        random_state=SEED,
    )


def predict_mean(models, X):
    """Average probability over fold models."""
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)
