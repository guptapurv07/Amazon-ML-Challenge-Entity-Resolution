"""Macro F0.5 over S1 entities, exactly as the leaderboard defines it.

Per S1 entity:
  truth empty & pred empty      -> 1.0
  exactly one of them empty     -> 0.0
  otherwise F0.5 = 1.25*P*R / (0.25*P + R)
The score is the plain mean over every S1 entity in the evaluation set.

Run `python src/metrics.py` to execute the unit tests.
"""
import machine_limits  # noqa: F401  -- must stay the first import (caps CPU threads)
import polars as pl

BETA2 = 0.25


def f05_single(pred, truth):
    """F0.5 for one entity given two iterables of ids."""
    pred, truth = set(pred), set(truth)
    if not pred and not truth:
        return 1.0
    tp = len(pred & truth)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(truth)
    return (1 + BETA2) * p * r / (BETA2 * p + r)


def per_entity_f05(pred_pairs, true_pairs, s1_ids):
    """Per-entity F0.5 table for every id in s1_ids.

    pred_pairs / true_pairs: DataFrames with columns (s1_id, match_id).
    """
    base = pl.DataFrame({"s1_id": s1_ids}, schema={"s1_id": pl.String})
    pred = pred_pairs.select("s1_id", "match_id").unique()
    truth = true_pairs.select("s1_id", "match_id").unique()
    n_pred = pred.group_by("s1_id").len("n_pred")
    n_true = truth.group_by("s1_id").len("n_true")
    tp = pred.join(truth, on=["s1_id", "match_id"]).group_by("s1_id").len("tp")
    df = (
        base.join(n_pred, on="s1_id", how="left")
        .join(n_true, on="s1_id", how="left")
        .join(tp, on="s1_id", how="left")
        .with_columns(pl.col("n_pred", "n_true", "tp").fill_null(0))
    )
    p = pl.col("tp") / pl.col("n_pred")
    r = pl.col("tp") / pl.col("n_true")
    f = (
        pl.when((pl.col("n_pred") == 0) & (pl.col("n_true") == 0)).then(1.0)
        .when(pl.col("tp") == 0).then(0.0)
        .otherwise((1 + BETA2) * p * r / (BETA2 * p + r))
    )
    return df.with_columns(f.alias("f05"))


def macro_f05(pred_pairs, true_pairs, s1_ids):
    return per_entity_f05(pred_pairs, true_pairs, s1_ids)["f05"].mean()


def _self_test():
    assert abs(f05_single(["S2-00047", "S2-00193", "S3-00812"],
                          ["S2-00047", "S3-00812"]) - 0.714) < 1e-3
    assert f05_single([], []) == 1.0
    assert f05_single(["S2-1"], []) == 0.0
    assert f05_single([], ["S2-1"]) == 0.0
    assert f05_single(["S2-1"], ["S2-2"]) == 0.0

    pairs = lambda rows: pl.DataFrame(rows, schema=["s1_id", "match_id"], orient="row")
    pred = pairs([("A", "S2-00047"), ("A", "S2-00193"), ("A", "S3-00812"), ("C", "S2-9")])
    truth = pairs([("A", "S2-00047"), ("A", "S3-00812"), ("D", "S2-5")])
    per = per_entity_f05(pred, truth, ["A", "B", "C", "D"])
    got = dict(zip(per["s1_id"], per["f05"]))
    # A: 0.714, B: singleton hit 1.0, C: false merge 0.0, D: miss 0.0
    assert abs(got["A"] - 0.7143) < 1e-3 and got["B"] == 1.0
    assert got["C"] == 0.0 and got["D"] == 0.0
    assert abs(macro_f05(pred, truth, ["A", "B", "C", "D"]) - (0.7143 + 1.0) / 4) < 1e-3
    print("metrics self-test passed (0.714 example OK)")


if __name__ == "__main__":
    _self_test()
