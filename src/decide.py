"""Decision layer: from scored candidate pairs to final match lists.

1. One-to-one: every S2/S3 record belongs to at most one S1 entity (verified
   on the train labels: no record appears under two S1), so each record is
   kept only for its highest-probability S1.
2a. Threshold: keep pairs with probability >= t, where t maximizes macro F0.5
    on out-of-fold predictions. An S1 left with no pairs is predicted singleton.
2b. Expected F0.5 (per S1): keep the k most probable candidates, choosing k
    (0 allowed) to maximize expected F0.5 under the model's probabilities.
Optional label-shift correction (prior_shift): test data can hold a different
share of decoys than train; the share of true pairs among each test country's
candidates is re-estimated with EM on the model's own probabilities, and the
probabilities are recalibrated to it before the threshold is applied.
"""
import machine_limits  # noqa: F401  -- must stay the first import (caps CPU threads)
import numpy as np
import polars as pl

from metrics import macro_f05

# fine steps at the top: on a country without training data the best threshold is high
THRESHOLD_GRID = np.round(np.concatenate([np.arange(0.20, 0.951, 0.025), [0.96, 0.97, 0.98, 0.99]]), 3)


def best_s1_per_record(pred):
    """pred: (s1_id, match_id, p). Keep each match_id only with its best S1."""
    return (pred.sort(["p", "s1_id"], descending=[True, False])
                .unique("match_id", keep="first", maintain_order=True))


def decide(pred, threshold):
    return best_s1_per_record(pred).filter(pl.col("p") >= threshold)


def decide_expected_f(pred, extra=0.0, beta2=0.25):
    """Per S1, keep the top-k candidates that maximize expected F0.5.

    With candidates sorted by p, choosing the top k gives approximately
      E[F] = (1 + b2) * sum_{i<=k} p_i / (b2 * E[#true] + k),
    E[#true] = sum of all p_i + `extra` (true matches missing from the candidates).
    Choosing nothing scores 1 only for a singleton: E[F] = P(no true match)
    = prod(1 - p_i) * exp(-extra).
    """
    one = best_s1_per_record(pred)
    ranked = one.sort(["s1_id", "p"], descending=[False, True]).with_columns(
        pl.int_range(1, pl.len() + 1).over("s1_id").alias("k"),
        pl.col("p").cum_sum().over("s1_id").alias("tp_k"),
        (pl.col("p").sum().over("s1_id") + extra).alias("n_true"),
        ((1 - pl.col("p")).clip(1e-12, 1).log().sum().over("s1_id") - extra).exp().alias("ef0"),
    ).with_columns(
        ((1 + beta2) * pl.col("tp_k") / (beta2 * pl.col("n_true") + pl.col("k"))).alias("ef"))
    best = ranked.group_by("s1_id").agg(
        pl.col("k").get(pl.col("ef").arg_max()).alias("best_k"),
        pl.col("ef").max().alias("best_ef"),
        pl.col("ef0").first(),
    ).with_columns(
        pl.when(pl.col("best_ef") > pl.col("ef0")).then(pl.col("best_k")).otherwise(0).alias("keep_k"))
    return (ranked.join(best.select("s1_id", "keep_k"), on="s1_id")
                  .filter(pl.col("k") <= pl.col("keep_k"))
                  .select(pred.columns))


def prior_shift(p, pi_train, max_iter=200, tol=1e-7):
    """Label-shift correction (Saerens, Latinne & Decaestecker, 2002).

    p: probabilities from a model trained where a share `pi_train` of the pairs
    were true. EM re-estimates that share on the new data (no labels needed)
    and returns (recalibrated probabilities, estimated share).
    """
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1 - 1e-6)
    pi = float(pi_train)
    for _ in range(max_iter):
        a = p * (pi / pi_train)
        q = a / (a + (1 - p) * ((1 - pi) / (1 - pi_train)))
        new = float(q.mean())
        if abs(new - pi) < tol:
            break
        pi = new
    return q, pi


def tune_threshold(pred, truth, s1_ids, grid=THRESHOLD_GRID):
    """Best threshold by macro F0.5. Returns (threshold, score, [(t, score), ...])."""
    one = best_s1_per_record(pred)
    curve = [(float(t), macro_f05(one.filter(pl.col("p") >= t), truth, s1_ids)) for t in grid]
    t, score = max(curve, key=lambda r: r[1])
    return t, score, curve
