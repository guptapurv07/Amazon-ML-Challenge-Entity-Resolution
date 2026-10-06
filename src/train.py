"""Train the pair classifier on train candidates and report validation scores.

Per train country, a deterministic SAMPLE_PCT% of S1 entities is used (all of
their candidates, labels from the ground truth). Reported:
  - Baseline 1 (rules): name/address token-set thresholds, best of a small grid
  - CV: folds grouped by S1 id (fold = hash(S1 id) % N_FOLDS, so all pairs of an
    S1 entity share a fold) -> out-of-fold probabilities -> macro F0.5 with the
    threshold tuned on those OOF predictions
  - LOCO: train on the other countries, score the held-out one (proxy for France)
The fold models, feature list and threshold are saved to work/model.joblib.
Macro F0.5 is always computed over ALL sampled S1 entities, including the ones
with no candidates (they are predicted empty).

Run from student_resource/:  python src/train.py [--no-loco] [--sample-pct 2]
"""
import machine_limits  # noqa: F401  -- must stay the first import (caps CPU threads)
import argparse

import joblib
import numpy as np
import polars as pl

from candidates import cand_paths, load_records
from data_io import WORK, list_countries, load_gt_pairs, load_norm, log
from decide import decide, prior_shift, tune_threshold
from features import CHUNK_PAIRS, pair_features, record_table
from metrics import macro_f05
from model import SEED, feature_columns, make_model, predict_mean, to_matrix
from te import name_diffs, oof_te_features, pseudo_te_features, te_features

SAMPLE_PCT = 25
N_FOLDS = 4
MODEL_PATH = WORK / "model.joblib"
LOCO_PATH = WORK / "loco.parquet"
TRAIN_X_PATH = WORK / "train_X.npy"  # feature matrix of the sample, for experiments without rebuilding
# columns p_<name> of LOCO_PATH, one per unseen-country variant
LOCO_COLS = {"with word scores": "with", "without word scores": "without",
             "self-trained word scores": "self"}


def is_sampled(col, pct=SAMPLE_PCT):
    return (pl.col(col).hash(seed=SEED) % 100) < pct


def build_training_table(country, gt, pct=SAMPLE_PCT):
    """Features + labels for the sampled S1 entities of one train country.
    Returns (table, sampled S1 ids, their true pairs)."""
    cand_path, ids_path = cand_paths("train", country)
    ids = pl.read_parquet(ids_path)
    s1 = ids.filter((pl.col("src") == "S1") & is_sampled("entity_id", pct))
    cands = pl.read_parquet(cand_path).join(
        s1.select(pl.col("rid").alias("s1_rid")), on="s1_rid", how="semi")
    df = load_records("train", country)
    assert df["entity_id"].equals(ids["entity_id"]), "record order changed since candidates"
    rec = record_table(df)
    del df
    feats = pl.concat([pair_features(cands.slice(i, CHUNK_PAIRS), rec)
                       for i in range(0, cands.height, CHUNK_PAIRS)])
    feats = pl.concat([feats, name_diffs(cands, rec)], how="horizontal")
    eid = ids["entity_id"]
    truth = gt.join(s1.select(pl.col("entity_id").alias("s1_id")), on="s1_id")
    feats = (
        feats.with_columns(
            eid.gather(feats["s1_rid"]).alias("s1_id"),
            eid.gather(feats["m_rid"]).alias("match_id"),
            pl.lit(country).alias("country"))
        .join(truth.with_columns(pl.lit(1, pl.Int8).alias("label")),
              on=["s1_id", "match_id"], how="left")
        .with_columns(pl.col("label").fill_null(0))
    )
    log(f"[{country}] {s1.height} sampled S1, {feats.height} pairs, "
        f"{int(feats['label'].sum())}/{truth.height} true pairs in candidates")
    return feats, s1["entity_id"], truth


def rule_baseline(data, truth, s1_ids):
    """Baseline 1: match when name and address token-set scores clear thresholds."""
    best = (None, -1.0)
    for tn in (85, 90, 95, 100):
        for ta in (0, 50, 70, 85):
            pred = data.filter((pl.col("name_tsr") >= tn) & (pl.col("addr_tsr") >= ta)).select(
                "s1_id", "match_id", (pl.col("name_tsr") + pl.col("addr_tsr")).alias("p"))
            score = macro_f05(decide(pred, 0.0), truth, s1_ids)
            if score > best[1]:
                best = ((tn, ta), score)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-loco", action="store_true", help="skip leave-one-country-out")
    ap.add_argument("--reuse-unseen-rule", action="store_true",
                    help="skip leave-one-country-out and keep the rule for countries without "
                         "training data chosen by the previous run (work/model.joblib)")
    ap.add_argument("--sample-pct", type=int, default=SAMPLE_PCT,
                    help="%% of train S1 entities to use (small values for a smoke test)")
    args = ap.parse_args()
    prev_unseen = joblib.load(MODEL_PATH)["unseen"] if args.reuse_unseen_rule else None
    gt = load_gt_pairs()
    countries = list_countries("train")
    tables, s1_sets, truths = [], {}, {}
    for country in countries:
        feats, s1_ids, truth = build_training_table(country, gt, args.sample_pct)
        tables.append(feats)
        s1_sets[country], truths[country] = s1_ids, truth
    data = pl.concat(tables, how="vertical_relaxed")
    del tables
    all_s1 = pl.concat(list(s1_sets.values()))
    all_truth = pl.concat(list(truths.values()))

    (tn, ta), rule_score = rule_baseline(data, all_truth, all_s1)
    log(f"Baseline 1 (rules: name_tsr>={tn}, addr_tsr>={ta}): macro F0.5 = {rule_score:.4f}")

    y = data["label"].to_numpy()
    # grouped folds without sklearn's GroupKFold: its shuffle mode runs np.isin on
    # string ids, which is quadratic for object arrays and never finishes here
    fold_of = data.select((pl.col("s1_id").hash(seed=SEED + 1) % N_FOLDS).alias("f"))["f"].to_numpy()
    diffs = data.select("extra", "missing")
    te, te_stats, te_prior = oof_te_features(diffs, y, fold_of, N_FOLDS)
    data = pl.concat([data.drop("extra", "missing"), te], how="horizontal")
    cols = feature_columns(data)
    te_cols = te.columns
    te_idx = [cols.index(c) for c in te_cols]
    X = to_matrix(data, cols)
    data = data.select("s1_id", "match_id", "country", "label")  # features now live in X
    del te
    log(f"training on {len(y)} pairs x {len(cols)} features, positive rate {y.mean():.3f}")
    oof, models = np.zeros(len(y)), []
    for fold in range(N_FOLDS):
        tr, va = fold_of != fold, fold_of == fold
        m = make_model().fit(X[tr], y[tr])
        oof[va] = m.predict_proba(X[va])[:, 1]
        models.append(m)
        log(f"  fold {fold}: {m.n_iter_} trees")
    pred = data.select("s1_id", "match_id", "country").with_columns(pl.Series("p", oof))
    # kept for decision-layer experiments without retraining
    pred.with_columns(data["label"]).write_parquet(WORK / "oof.parquet")
    np.save(TRAIN_X_PATH, X)  # rows as in oof.parquet, columns = the bundle's "features"
    pl.concat([pl.DataFrame({"s1_id": s, "country": c}) for c, s in s1_sets.items()],
              how="vertical_relaxed").write_parquet(WORK / "oof_s1.parquet")
    t, cv_score, curve = tune_threshold(pred.drop("country"), all_truth, all_s1)
    log(f"CV macro F0.5 = {cv_score:.4f} at threshold {t:.3f}")
    log("  threshold curve: " + ", ".join(f"{a:.2f}:{b:.4f}" for a, b in curve[::4]))
    per_country = {}
    for country in countries:
        per_country[country] = macro_f05(
            decide(pred.filter(pl.col("country") == country).drop("country"), t),
            truths[country], s1_sets[country])
        log(f"  CV {country}: {per_country[country]:.4f}")
    # the leaderboard mixes countries by their share of TEST S1 entities
    test_n = {c: load_norm("test", c, columns=["src"]).filter(pl.col("src") == "S1").height
              for c in list_countries("test")}
    seen = sum(test_n[c] for c in countries if c in test_n)
    test_mix = sum(per_country[c] * test_n.get(c, 0) for c in countries) / seen
    unseen_share = 1 - seen / sum(test_n.values())
    log(f"CV in test country mix: {test_mix:.4f} (countries without training data: "
        f"{unseen_share:.1%} of test S1)")

    # Countries without training data (France): the model and threshold are chosen by
    # leave-one-country-out. Word scores come from the training countries only, so the
    # held-out country's words look unseen, as French words do.
    unseen = {"variant": "with word scores", "threshold": t, "model": None, "features": cols}
    if prev_unseen is not None:
        unseen["variant"], unseen["threshold"] = prev_unseen["variant"], prev_unseen["threshold"]
        if unseen["variant"] in ("without word scores", "self-trained word scores"):
            agnostic = [c for c in cols if not c.startswith("te_")]
            X = X[:, [cols.index(c) for c in agnostic]]
            unseen.update(model=make_model().fit(X, y), features=agnostic,
                          self_trained=unseen["variant"] == "self-trained word scores")
        log(f"countries without training data: previous rule kept ({unseen['variant']}, "
            f"threshold {unseen['threshold']:.3f})")
    elif not args.no_loco and len(countries) > 1:
        agnostic = [c for c in cols if not c.startswith("te_")]  # no word identities
        ag_idx = [cols.index(c) for c in agnostic]
        variants = {"with word scores": {}, "without word scores": {}, "self-trained word scores": {}}
        loco = []  # raw held-out probabilities, for a stage 2 fitted to unseen countries (stage2.py)
        for held in countries:
            tr = (data["country"] != held).to_numpy()
            te_tr, stats_tr, prior_tr = oof_te_features(
                diffs.filter(pl.Series(tr)), y[tr], fold_of[tr], N_FOLDS)
            X_tr, X_ho = X[tr], X[~tr]
            X_tr[:, te_idx] = to_matrix(te_tr, te_cols)
            X_ho[:, te_idx] = to_matrix(
                te_features(diffs.filter(~pl.Series(tr)), stats_tr, prior_tr), te_cols)
            m_full = make_model().fit(X_tr, y[tr])
            p_full = m_full.predict_proba(X_ho)[:, 1]
            X_tr = X_tr[:, ag_idx]
            p_ag = make_model().fit(X_tr, y[tr]).predict_proba(X_ho[:, ag_idx])[:, 1]
            del X_tr
            # self-trained word scores: statistics from the held-out country's own confident
            # pairs (rated by the model without word scores), then the full model
            X_ho[:, te_idx] = to_matrix(pseudo_te_features(
                diffs.filter(~pl.Series(tr)), p_ag, fold_of[~tr], N_FOLDS), te_cols)
            p_self = m_full.predict_proba(X_ho)[:, 1]
            del X_ho, m_full
            ids = data.filter(~pl.Series(tr)).select("s1_id", "match_id")
            held_out = (("with word scores", p_full), ("without word scores", p_ag),
                        ("self-trained word scores", p_self))
            loco.append(ids.with_columns(pl.lit(held).alias("country"), pl.Series("label", y[~tr]),
                                         *[pl.Series(f"p_{LOCO_COLS[n]}", p) for n, p in held_out]))
            for name, p_ho in held_out:
                q, pi = prior_shift(p_ho, float(y[tr].mean()))  # as predict.py does
                variants[name][held] = ids.with_columns(pl.Series("p", q))
                bt, bs, _ = tune_threshold(variants[name][held], truths[held], s1_sets[held])
                log(f"LOCO held-out {held}, {name}: "
                    f"{macro_f05(decide(variants[name][held], t), truths[held], s1_sets[held]):.4f} "
                    f"at CV threshold, {bs:.4f} at its best {bt:.3f} (prior shift share {pi:.3f})")
        pl.concat(loco).write_parquet(LOCO_PATH)
        del loco
        best_est = -1.0
        for name, preds in variants.items():
            thr = float(np.mean([tune_threshold(preds[c], truths[c], s1_sets[c])[0] for c in countries]))
            for th in (t, thr):
                est = float(np.mean([macro_f05(decide(preds[c], th), truths[c], s1_sets[c])
                                     for c in countries]))
                log(f"unseen countries {name}, threshold {th:.3f}: mean LOCO {est:.4f} -> "
                    f"leaderboard estimate {(1 - unseen_share) * test_mix + unseen_share * est:.4f}")
                if est > best_est:
                    best_est, unseen = est, {"variant": name, "threshold": th, "model": None,
                                             "features": cols}
        if unseen["variant"] in ("without word scores", "self-trained word scores"):
            X = X[:, ag_idx]  # the model without word scores (it also rates pairs for self-training)
            unseen.update(model=make_model().fit(X, y), features=agnostic,
                          self_trained=unseen["variant"] == "self-trained word scores")
        log(f"chosen for countries without training data: {unseen['variant']}, threshold "
            f"{unseen['threshold']:.3f} (tuned on the held-out countries: slightly optimistic)")

    joblib.dump({"models": models, "features": cols, "threshold": t,
                 "unseen": unseen, "train_countries": countries,
                 "pi_train": float(y.mean()), "te_stats": te_stats, "te_prior": te_prior},
                MODEL_PATH)
    log(f"saved {MODEL_PATH.name}")


if __name__ == "__main__":
    main()
