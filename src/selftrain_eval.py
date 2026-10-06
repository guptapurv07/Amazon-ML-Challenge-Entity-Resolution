"""Leave-one-country-out check of SELF-TRAINING the pair model for countries without
training data (France).

The model without word scores is what countries without training data get today.
Self-training refits it on the train countries' labeled pairs PLUS the new
country's own confident pairs, rated by that same model (probability <= LO -> 0,
>= HI -> 1; the labels come from the model, never by hand), so the trees also learn
how the new country's names and addresses look.

Check, for each train country c: the model fitted on the other countries rates c's
pairs (p_without in loco.parquet, written by train.py); the self-trained model is
fitted on the other countries' pairs plus c's confident pairs, then rates all of c's
pairs. Both get the same EM prior shift and one-to-one decision. Reported: macro F0.5
on c at the CV threshold and at c's best threshold, and the mean over countries at
the mean best threshold (the leaderboard proxy train.py uses for France).

Needs work/train_X.npy, oof.parquet, oof_s1.parquet, loco.parquet and model.joblib
from one train.py run with leave-one-country-out.
Run from student_resource/:  python src/selftrain_eval.py
"""
import machine_limits  # noqa: F401  -- must stay the first import (caps CPU threads)
import joblib
import numpy as np
import polars as pl

from data_io import WORK, load_gt_pairs, log
from decide import decide, prior_shift, tune_threshold
from metrics import macro_f05
from model import make_model
from train import LOCO_PATH, MODEL_PATH, TRAIN_X_PATH

LO, HI = 0.03, 0.97  # confident pairs, as in te.pseudo_te_features


def main():
    bundle = joblib.load(MODEL_PATH)
    agnostic = [i for i, c in enumerate(bundle["features"]) if not c.startswith("te_")]
    X = np.load(TRAIN_X_PATH, mmap_mode="r")
    oof = pl.read_parquet(WORK / "oof.parquet").with_row_index("row")
    assert X.shape == (oof.height, len(bundle["features"])), "train_X.npy is from another run"
    y = oof["label"].to_numpy()
    loco = pl.read_parquet(LOCO_PATH, columns=["s1_id", "match_id", "p_without"])
    s1 = pl.read_parquet(WORK / "oof_s1.parquet")
    gt = load_gt_pairs()
    t = bundle["threshold"]
    countries = oof["country"].unique().sort().to_list()
    preds, truths, s1_sets = {"without word scores": {}, "self-trained model": {}}, {}, {}
    for c in countries:
        held = oof.filter(pl.col("country") == c).join(loco, on=["s1_id", "match_id"], how="left")
        assert held["p_without"].null_count() == 0, "loco.parquet is from another run"
        rows_c = held["row"].to_numpy()
        rows_d = np.flatnonzero((oof["country"] != c).to_numpy())
        p_ag = held["p_without"].to_numpy()
        sure = (p_ag <= LO) | (p_ag >= HI)
        Xc = X[np.ix_(rows_c, agnostic)]
        fit_X = np.concatenate([X[np.ix_(rows_d, agnostic)], Xc[sure]])
        fit_y = np.concatenate([y[rows_d], (p_ag[sure] >= HI).astype(y.dtype)])
        log(f"[{c}] self-training on {len(rows_d)} labeled pairs of the other countries + "
            f"{sure.sum()} confident pairs of {c} ({(p_ag[sure] >= HI).mean():.3f} rated true); "
            f"their true precision {y[rows_c][sure][p_ag[sure] >= HI].mean():.4f}, "
            f"false rate among rated false {y[rows_c][sure][p_ag[sure] <= LO].mean():.4f}")
        model = make_model().fit(fit_X, fit_y)
        del fit_X, fit_y
        p_st = model.predict_proba(Xc)[:, 1]
        del Xc, model
        pi = float(y[rows_d].mean())
        s1_sets[c] = s1.filter(pl.col("country") == c)["s1_id"]
        truths[c] = gt.join(s1_sets[c].to_frame("s1_id"), on="s1_id")
        for name, p in (("without word scores", p_ag), ("self-trained model", p_st)):
            q, share = prior_shift(p, pi)
            preds[name][c] = held.select("s1_id", "match_id").with_columns(pl.Series("p", q))
            bt, bs, _ = tune_threshold(preds[name][c], truths[c], s1_sets[c])
            log(f"[{c}] {name}: {macro_f05(decide(preds[name][c], t), truths[c], s1_sets[c]):.4f} "
                f"at CV threshold {t:.3f}, {bs:.4f} at its best {bt:.3f} (prior shift share {share:.3f})")
    for name, pr in preds.items():
        thr = float(np.mean([tune_threshold(pr[c], truths[c], s1_sets[c])[0] for c in countries]))
        per = [macro_f05(decide(pr[c], thr), truths[c], s1_sets[c]) for c in countries]
        log(f"{name}, threshold {thr:.3f}: mean leave-one-country-out {np.mean(per):.4f} "
            f"({', '.join(f'{c} {v:.4f}' for c, v in zip(countries, per))})")


if __name__ == "__main__":
    main()
