"""Stage 2: combine the pair model with the cross-encoder on the hard pairs.

The pair model (stage 1) is confident on most pairs. For the hard pairs (stage-1
probability in [LO, HI], see ce_export.py) a multilingual cross-encoder,
fine-tuned on OTHER train entities (colab/cross_encoder.ipynb), gives a second
opinion. Stage 2 is a small gradient-boosted model on
  z1         logit of the stage-1 probability
  ce         cross-encoder logit
  ce_gap_s1  ce minus the best ce among the S1 entity's hard pairs
  ce_margin_m  ce minus the best ce of the same S2/S3 record under another S1
  n_hard_s1 / n_hard_m  number of hard pairs of the S1 entity / of the record
trained on the hard pairs of the pair-model sample (out-of-fold stage-1
probability, same S1-grouped folds as train.py). Its out-of-fold probabilities
replace stage 1 on the hard pairs; every other pair keeps its stage-1 probability.
Reported: CV macro F0.5 before and after, per country and in the test mix.

Test: the same replacement, then the usual EM prior shift and one-to-one decision.
Countries without training data keep the rule train.py chose (their own model and
threshold); --stage2-unseen applies stage 2 to them as well (z1 from their model).

--loco-stage2 checks that choice by leave-one-country-out (train.py saves the
held-out probabilities in loco.parquet; ce_export.py --round2 score exports their hard
pairs for the cross-encoder): for each train country, a stage 2 fitted on the OTHER
country only is applied to the country's held-out pairs, either the seen countries'
stage 2 (what --stage2-unseen does) or an own stage 2 fitted on held-out pairs (z1
from a model that never saw the country, as for France, and as many hard pairs). The
better one by mean held-out F0.5, with its threshold, is used for countries without
training data. The cross-encoder itself saw both train countries, so these numbers
are optimistic for France; they rank the two stage 2 variants.

Needs work/oof.parquet, work/oof_s1.parquet, work/test_predictions.parquet (stage 1)
and work/ce/ce_scores_oof.parquet, work/ce/ce_scores_test.parquet (Colab).
Run from student_resource/:  python src/stage2.py [--stage2-unseen] [--out-dir work/sub_stage2]
"""
import machine_limits  # noqa: F401  -- must stay the first import (caps CPU threads)
import argparse
from pathlib import Path

import joblib
import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from candidates import cand_paths, load_records
from ce_export import CE_DIR, HI, LO
from data_io import ROOT, WORK, list_countries, load_gt_pairs, load_norm, log, write_id_lists
from decide import decide, prior_shift, tune_threshold
from features import CHUNK_PAIRS, pair_features, record_table
from metrics import macro_f05
from model import SEED
from predict import test_s1_ids, unseen_rule
from train import LOCO_COLS, LOCO_PATH, MODEL_PATH, N_FOLDS, TRAIN_X_PATH

FEATURES = ["z1", "ce", "ce_gap_s1", "ce_margin_m", "n_hard_s1", "n_hard_m"]
# --pair-feats: a few pair features next to the cross-encoder, so stage 2 can overrule it
# where the text looks alike but the numbers say otherwise (a look-alike a few doors away)
EXTRA = ["name_tsr", "addr_tsr", "core_ratio", "num_conflict", "num_first_eq", "num_first_ratio",
         "num_first_diff", "num_min_shift", "legal_conflict", "is_s2", "m_addr_empty"]
STAGE2_PATH = WORK / "stage2.joblib"


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def make_stage2():
    return HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_leaf_nodes=15,
                                          min_samples_leaf=100, l2_regularization=1.0,
                                          early_stopping=True, validation_fraction=0.1,
                                          n_iter_no_change=30, random_state=SEED)


def stage2_table(hard, p_col):
    """Stage-2 features of hard pairs (s1_id, match_id, p_col, ce)."""
    best_m = pl.col("ce").max().over("match_id")
    second_m = pl.col("ce").top_k(2).min().over("match_id")
    other_m = pl.when(pl.col("ce") >= best_m).then(second_m).otherwise(best_m)
    return hard.with_columns(
        pl.Series("z1", logit(hard[p_col].to_numpy())),
        (pl.col("ce") - pl.col("ce").max().over("s1_id")).alias("ce_gap_s1"),
        pl.when(pl.len().over("match_id") > 1).then(pl.col("ce") - other_m)
          .otherwise(20.0).alias("ce_margin_m"),
        pl.len().over("s1_id").cast(pl.Float64).alias("n_hard_s1"),
        pl.len().over("match_id").cast(pl.Float64).alias("n_hard_m"),
    ).with_columns(ce_b_features() if "ce_b" in hard.columns else [])


def ce_b_features():
    """Gap within the S1 and margin within the record for a second cross-encoder (ce_b)."""
    best_m = pl.col("ce_b").max().over("match_id")
    second_m = pl.col("ce_b").top_k(2).min().over("match_id")
    other_m = pl.when(pl.col("ce_b") >= best_m).then(second_m).otherwise(best_m)
    return [(pl.col("ce_b") - pl.col("ce_b").max().over("s1_id")).alias("ce_b_gap_s1"),
            pl.when(pl.len().over("match_id") > 1).then(pl.col("ce_b") - other_m)
              .otherwise(20.0).alias("ce_b_margin_m")]


def with_ce(pairs, name, ce_dir=CE_DIR, prefix="ce", col="ce"):
    """Attach a cross-encoder logit (column `col`; null where the pair was not scored)."""
    files = sorted(Path(ce_dir).glob(f"{prefix}_scores_{name}*.parquet"))  # + top-up files
    ce = pl.concat([pl.read_parquet(f) for f in files]).unique(["s1_id", "match_id"]).select(
        "s1_id", "match_id", pl.col("ce_logit").cast(pl.Float64).alias(col))
    return pairs.join(ce, on=["s1_id", "match_id"], how="left")


def add_sample_extra(tab, oof_rows, stage1, bundle):
    """EXTRA columns for pairs of the pair-model sample (oof_rows = their row in oof.parquet),
    from the feature matrix train.py saved."""
    path = stage1 / TRAIN_X_PATH.name if (stage1 / TRAIN_X_PATH.name).exists() else TRAIN_X_PATH
    X = np.load(path, mmap_mode="r")
    assert X.shape[1] == len(bundle["features"]), f"{path} is not from this stage 1"
    rows = oof_rows.to_numpy()
    return tab.with_columns([pl.Series(c, np.asarray(X[rows, bundle["features"].index(c)], np.float64))
                             for c in EXTRA])


def test_extra(pairs, country):
    """EXTRA columns for test pairs (s1_id, match_id) of one country, recomputed from the
    candidate table exactly as predict.py computes them."""
    cand_path, ids_path = cand_paths("test", country)
    ids = pl.read_parquet(ids_path, columns=["rid", "entity_id"])
    keys = (pairs.select("s1_id", "match_id")
                 .join(ids.rename({"rid": "s1_rid", "entity_id": "s1_id"}), on="s1_id")
                 .join(ids.rename({"rid": "m_rid", "entity_id": "match_id"}), on="match_id"))
    cands = pl.read_parquet(cand_path).join(keys, on=["s1_rid", "m_rid"], how="inner")
    df = load_records("test", country)
    rec = record_table(df)
    del df
    return pl.concat([pair_features(cands.slice(i, CHUNK_PAIRS), rec).select(
        "s1_id", "match_id", *[pl.col(c).cast(pl.Float64) for c in EXTRA])
        for i in range(0, cands.height, CHUNK_PAIRS)])


def with_p2(pairs, hard_tab, model, p_col):
    """pairs with p_final = stage-2 probability on hard_tab's pairs, else p_col."""
    p2 = hard_tab.select("s1_id", "match_id",
                         pl.Series("p2", model.predict_proba(hard_tab.select(FEATURES).to_numpy())[:, 1]))
    return pairs.join(p2, on=["s1_id", "match_id"], how="left").with_columns(
        pl.coalesce("p2", p_col).alias("p_final"))


def unseen_stage2(loco, col, hard, final, t2, s1, truth):
    """Leave-one-country-out choice of the stage 2 for countries without training data
    (see the module docstring). loco: held-out pairs (country, label, col, ce); hard:
    the seen countries' stage-2 table. Returns {name, model, threshold}."""
    lhard = stage2_table(loco.filter(pl.col(col).is_between(LO, HI) & pl.col("ce").is_not_null()), col)
    n_band = loco.filter(pl.col(col).is_between(LO, HI)).height
    log(f"leave-one-country-out: {n_band} hard pairs ({col}), {lhard.height} with a cross-encoder score"
        + ("" if lhard.height >= 0.99 * n_band else "  <- missing scores: upload ce2_oof_loco"))
    lX, ly = lhard.select(FEATURES).to_numpy(), lhard["label"].to_numpy()
    X, y = hard.select(FEATURES).to_numpy(), hard["label"].to_numpy()
    countries = loco["country"].unique().sort().to_list()
    s1_sets = {c: s1.filter(pl.col("country") == c)["s1_id"] for c in countries}
    truths = {c: truth.join(s1_sets[c].to_frame("s1_id"), on="s1_id") for c in countries}
    preds = {"stage 1 only": {}, "seen countries' stage 2": {}, "own stage 2": {}}
    for c in countries:
        base = loco.filter(pl.col("country") == c).select("s1_id", "match_id", col)
        held = lhard.filter(pl.col("country") == c)
        pi = float(loco.filter(pl.col("country") != c)["label"].mean())
        other_m, other_l = (hard["country"] != c).to_numpy(), (lhard["country"] != c).to_numpy()
        for name, model in (("stage 1 only", None),
                            ("seen countries' stage 2", make_stage2().fit(X[other_m], y[other_m])),
                            ("own stage 2", make_stage2().fit(lX[other_l], ly[other_l]))):
            d = base.with_columns(pl.col(col).alias("p_final")) if model is None else with_p2(base, held, model, col)
            q, _ = prior_shift(d["p_final"].to_numpy(), pi)  # as at test time
            preds[name][c] = d.select("s1_id", "match_id").with_columns(pl.Series("p", q))
    best = None
    for name, pr in preds.items():
        thr = float(np.mean([tune_threshold(pr[c], truths[c], s1_sets[c])[0] for c in countries]))
        for th in sorted({thr, t2}):
            per = [macro_f05(decide(pr[c], th), truths[c], s1_sets[c]) for c in countries]
            log(f"  unseen countries, {name}, threshold {th:.3f}: mean leave-one-country-out "
                f"{np.mean(per):.4f} ({', '.join(f'{c} {v:.4f}' for c, v in zip(countries, per))})")
            # stage 1 only is reported, not chosen: the cross-encoder saw both train countries,
            # so this check cannot say how much it helps a truly unseen country
            if name != "stage 1 only" and (best is None or np.mean(per) > best[2]):
                best = (name, th, float(np.mean(per)))
    model = make_stage2().fit(lX, ly) if best[0] == "own stage 2" else final
    log(f"chosen for countries without training data: {best[0]}, threshold {best[1]:.3f}")
    return {"name": best[0], "model": model, "threshold": best[1]}


def main():
    global FEATURES
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage2-unseen", action="store_true",
                    help="also apply stage 2 to countries without training data")
    ap.add_argument("--loco-stage2", action="store_true",
                    help="stage 2 for countries without training data chosen by leave-one-country-out "
                         "(needs loco.parquet in the stage-1 folder; implies --stage2-unseen)")
    ap.add_argument("--out-dir", default="work/sub_stage2", help="where to write the two TSV files")
    ap.add_argument("--ce-dir", default=str(CE_DIR), help="folder with ce_scores_{oof,test}.parquet")
    ap.add_argument("--ce-prefix", default="ce",
                    help="ce = round-1 scores (ce_scores_*), ce2 = round-2 scores (ce2_scores_*)")
    ap.add_argument("--ce-unseen-prefix", default=None,
                    help="countries without training data: use this prefix's test scores where they "
                         "exist (ce3 = colab/cross_encoder_round3.ipynb), the --ce-prefix ones elsewhere")
    ap.add_argument("--ce-unseen-weight", type=float, default=1.0,
                    help="with --ce-unseen-prefix: cross-encoder logit = weight x that prefix + (1 - weight) x "
                         "--ce-prefix (0.5 = average of the two rounds)")
    ap.add_argument("--stage1-dir", default=str(WORK),
                    help="folder with the stage-1 model.joblib, oof.parquet, oof_s1.parquet, "
                         "test_predictions.parquet (a saved copy lets a new stage 1 train meanwhile)")
    ap.add_argument("--save-test-probs", action="store_true",
                    help="also write the final test probabilities (after prior shift) to OUT_DIR/test_probs.parquet")
    ap.add_argument("--ce-extra-prefix", default=None,
                    help="a second cross-encoder (e.g. cel = kaggle/cross_encoder_large.ipynb) as extra "
                         "stage-2 features next to --ce-prefix")
    ap.add_argument("--pair-feats", action="store_true",
                    help="give stage 2 the EXTRA pair features too (needs train_X.npy of this stage 1)")
    args = ap.parse_args()
    if args.pair_feats:
        FEATURES = FEATURES + EXTRA
    if args.ce_extra_prefix:
        FEATURES = FEATURES + ["ce_b", "ce_b_gap_s1", "ce_b_margin_m"]

    def extra(pairs, name):  # second cross-encoder, if any
        return with_ce(pairs, name, args.ce_dir, args.ce_extra_prefix, "ce_b") if args.ce_extra_prefix else pairs
    out_dir = Path(args.out_dir) if Path(args.out_dir).is_absolute() else ROOT / args.out_dir
    stage1 = Path(args.stage1_dir)
    bundle = joblib.load(stage1 / MODEL_PATH.name)

    # ---- train and validate on the pair-model sample (out-of-fold everywhere)
    oof = extra(with_ce(pl.read_parquet(stage1 / "oof.parquet").with_row_index("row"), "oof", args.ce_dir,
                        args.ce_prefix), "oof")
    s1 = pl.read_parquet(stage1 / "oof_s1.parquet")
    truth = load_gt_pairs().join(s1.select("s1_id"), on="s1_id")
    is_hard = pl.col("p").is_between(LO, HI) & pl.col("ce").is_not_null()
    hard = stage2_table(oof.filter(is_hard), "p")
    if args.pair_feats:
        hard = add_sample_extra(hard, hard["row"], stage1, bundle)
    log(f"{hard.height} hard pairs of {oof.height} ({hard.height / oof.height:.1%}), "
        f"positive rate {hard['label'].mean():.3f}")
    X, y = hard.select(FEATURES).to_numpy(), hard["label"].to_numpy()
    fold_of = hard.select((pl.col("s1_id").hash(seed=SEED + 1) % N_FOLDS).alias("f"))["f"].to_numpy()
    p2 = np.zeros(len(y))
    for fold in range(N_FOLDS):
        m = make_stage2().fit(X[fold_of != fold], y[fold_of != fold])
        p2[fold_of == fold] = m.predict_proba(X[fold_of == fold])[:, 1]
    final = make_stage2().fit(X, y)
    for name, score in (("stage 1", hard["p"].to_numpy()), ("cross-encoder", hard["ce"].to_numpy()),
                        ("stage 2", p2)):
        log(f"  AUC on hard pairs, {name}: {roc_auc_score(y, score):.4f}")
    combined = oof.join(hard.select("s1_id", "match_id", pl.Series("p2", p2)),
                        on=["s1_id", "match_id"], how="left").with_columns(
        pl.coalesce("p2", "p").alias("p_final"))
    combined.select("s1_id", "match_id", "country", "label", "p", "ce", "p_final").write_parquet(
        WORK / "stage2_oof.parquet")  # for error analysis
    t1, f1, _ = tune_threshold(oof.select("s1_id", "match_id", "p"), truth, s1["s1_id"])
    t2, f2, curve = tune_threshold(combined.select("s1_id", "match_id", pl.col("p_final").alias("p")),
                                   truth, s1["s1_id"])
    log(f"CV macro F0.5: stage 1 {f1:.4f} (threshold {t1:.3f}) -> stage 2 {f2:.4f} (threshold {t2:.3f})")
    test_n = {c: load_norm("test", c, columns=["src"]).filter(pl.col("src") == "S1").height
              for c in list_countries("test")}
    mix, weight = 0.0, 0
    for country in s1["country"].unique().sort().to_list():
        ids = s1.filter(pl.col("country") == country)["s1_id"]
        tr = truth.join(ids.to_frame("s1_id"), on="s1_id")
        c = combined.filter(pl.col("country") == country)
        a = macro_f05(decide(c.select("s1_id", "match_id", "p"), t1), tr, ids)
        b = macro_f05(decide(c.select("s1_id", "match_id", pl.col("p_final").alias("p")), t2), tr, ids)
        log(f"  CV {country}: {a:.4f} -> {b:.4f}")
        mix, weight = mix + b * test_n.get(country, 0), weight + test_n.get(country, 0)
    log(f"CV in test country mix after stage 2: {mix / weight:.4f}")
    joblib.dump({"model": final, "features": FEATURES, "threshold": t2}, STAGE2_PATH)

    unseen_s2 = None
    if args.loco_stage2:
        loco_path = stage1 / LOCO_PATH.name
        assert loco_path.exists(), f"{loco_path} missing: run train.py with leave-one-country-out"
        col = f"p_{LOCO_COLS[bundle['unseen']['variant']]}"
        loco = extra(with_ce(pl.read_parquet(loco_path), "oof", args.ce_dir, args.ce_prefix), "oof")
        if args.pair_feats:
            loco = loco.join(oof.select("s1_id", "match_id", "row"), on=["s1_id", "match_id"], how="left")
            loco = add_sample_extra(loco, loco["row"], stage1, bundle)
        unseen_s2 = unseen_stage2(loco, col, hard, final, t2, s1, truth)
        del loco

    # ---- test
    pred = extra(with_ce(pl.read_parquet(stage1 / "test_predictions.parquet"), "test", args.ce_dir,
                         args.ce_prefix), "test")
    if args.ce_unseen_prefix:
        pred = with_ce(pred.rename({"ce": "ce_main"}), "test", args.ce_dir, args.ce_unseen_prefix).rename(
            {"ce": "ce_unseen"}).rename({"ce_main": "ce"})
    parts, probs = [], []
    for country in list_countries("test"):
        sub = pred.filter(pl.col("country") == country)
        rule = unseen_rule(bundle, country)
        if rule is not None and args.ce_unseen_prefix:
            log(f"[{country}] {args.ce_unseen_prefix} scores for {sub['ce_unseen'].is_not_null().sum()} of "
                f"{sub['ce'].is_not_null().sum()} pairs with a {args.ce_prefix} score")
            w = args.ce_unseen_weight
            sub = sub.with_columns(pl.when(pl.col("ce_unseen").is_not_null() & pl.col("ce").is_not_null())
                                   .then(w * pl.col("ce_unseen") + (1 - w) * pl.col("ce"))
                                   .otherwise(pl.coalesce("ce_unseen", "ce")).alias("ce"))
        p_col = "p_unseen" if rule and rule["model"] is not None and "p_unseen" in sub.columns else "p"
        model, thr, how = final, t2, "stage 2"
        if rule is not None:  # no training data
            if unseen_s2 is not None:
                model, thr, how = unseen_s2["model"], unseen_s2["threshold"], unseen_s2["name"]
            elif not args.stage2_unseen:
                model, thr, how = None, rule["threshold"], "stage 1 only"
        d = sub.select("s1_id", "match_id", p_col).with_columns(pl.col(p_col).alias("p_final"))
        h = sub.filter(pl.col(p_col).is_between(LO, HI) & pl.col("ce").is_not_null())
        if model is not None and h.height:
            ht = stage2_table(h, p_col)
            if args.pair_feats:
                ht = ht.join(test_extra(ht, country), on=["s1_id", "match_id"], how="left")
            d = with_p2(sub.select("s1_id", "match_id", p_col), ht, model, p_col)
        q, pi = prior_shift(d["p_final"].to_numpy(), bundle["pi_train"])
        log(f"[{country}] {how} ({h.height if model is not None else 0} hard pairs, {p_col}), "
            f"prior shift share {pi:.4f}, threshold {thr:.3f}")
        parts.append(decide(d.select("s1_id", "match_id").with_columns(pl.Series("p", q)), thr))
        if args.save_test_probs:
            probs.append(d.select("s1_id", "match_id").with_columns(pl.Series("q", q), pl.lit(thr).alias("thr")))
    matches = pl.concat(parts)
    s1_test = test_s1_ids()
    write_id_lists(pred.select("s1_id", "match_id"), s1_test["entity_id"],
                   out_dir / "candidate_pairs.tsv", "candidate_entity_ids")
    write_id_lists(matches, s1_test["entity_id"], out_dir / "matching_results.tsv", "matched_entity_ids")
    log(f"wrote {out_dir}: {matches.height} matches from {pred.height} candidates")
    if probs:
        pl.concat(probs).write_parquet(out_dir / "test_probs.parquet")


if __name__ == "__main__":
    main()
