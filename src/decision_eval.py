"""Compare decision rules on the saved out-of-fold predictions of train.py.

Reads work/oof.parquet and work/oof_s1.parquet (no retraining needed).
Run from student_resource/:  python src/decision_eval.py
"""
import machine_limits  # noqa: F401  -- must stay the first import (caps CPU threads)
import polars as pl

from data_io import WORK, load_gt_pairs, log
from decide import decide, decide_expected_f, prior_shift, tune_threshold
from metrics import macro_f05


def main():
    oof = pl.read_parquet(WORK / "oof.parquet")
    s1 = pl.read_parquet(WORK / "oof_s1.parquet")
    truth = load_gt_pairs().join(s1.select("s1_id"), on="s1_id")
    pi_train = oof["label"].mean()
    for country in [None] + sorted(s1["country"].unique().to_list()):
        sel_s1 = s1 if country is None else s1.filter(pl.col("country") == country)
        pred = (oof if country is None else oof.filter(pl.col("country") == country)).select(
            "s1_id", "match_id", "p")
        tr = truth.join(sel_s1.select("s1_id"), on="s1_id")
        name = country or "ALL"
        t, best, _ = tune_threshold(pred, tr, sel_s1["s1_id"])
        log(f"[{name}] best global threshold {t:.3f}: {best:.4f}")
        for extra in (0.0, 0.1, 0.25, 0.5):
            sc = macro_f05(decide_expected_f(pred, extra=extra), tr, sel_s1["s1_id"])
            log(f"[{name}] expected-F0.5 rule, extra={extra}: {sc:.4f}")
        # EM must recover the known share of true pairs, or the correction can't be trusted
        q, pi_hat = prior_shift(pred["p"].to_numpy(), pi_train)
        true_share = (oof if country is None else oof.filter(pl.col("country") == country))["label"].mean()
        sc = macro_f05(decide(pred.with_columns(pl.Series("p", q)), t), tr, sel_s1["s1_id"])
        log(f"[{name}] prior shift: EM share {pi_hat:.4f} vs true {true_share:.4f} "
            f"(train {pi_train:.4f}); F0.5 at threshold {t:.3f} after shift: {sc:.4f}")


if __name__ == "__main__":
    main()
