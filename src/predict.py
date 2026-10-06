"""Score the test candidates and write both submission files.

  <out-dir>/candidate_pairs.tsv   every pair the model scored (the final candidate set)
  <out-dir>/matching_results.tsv  pairs kept by the decision layer (decide.py)
Every test S1 entity gets exactly one row in each file (empty list if none).
Scores are cached in work/test_predictions.parquet, so a different decision rule
can be written with --reuse-scores without scoring again.

Countries without training data (France) follow the rule train.py chose by
leave-one-country-out (bundle["unseen"]): possibly a model without the word
scores (column p_unseen) and its own threshold. --no-unseen-rule treats them
like the training countries.

Run from student_resource/:
  python src/predict.py                        # score + write output/ (with prior shift)
  python src/predict.py --reuse-scores --no-prior-shift --out-dir work/sub_plain
  python src/predict.py --reuse-scores --no-unseen-rule --out-dir work/sub_same_rule
The EM prior shift is on by default: on the leaderboard it beat the plain
threshold (0.958 vs 0.957, 2026-09-26), since test holds more decoys than train.
"""
import machine_limits  # noqa: F401  -- must stay the first import (caps CPU threads)
import argparse
from pathlib import Path

import joblib
import numpy as np
import polars as pl

from candidates import cand_paths, load_records
from data_io import OUTPUT, ROOT, WORK, list_countries, log, write_id_lists
from decide import decide, prior_shift
from features import CHUNK_PAIRS, pair_features, record_table
from model import SEED, predict_mean, to_matrix
from te import SIDES, name_diffs, pseudo_te_features, te_features
from train import MODEL_PATH, N_FOLDS

PRED_PATH = WORK / "test_predictions.parquet"


def unseen_rule(bundle, country):
    """The rule for a country without training data, or None for a training country."""
    if "unseen" in bundle and country not in bundle["train_countries"]:
        return bundle["unseen"]
    return None


def score_country(country, bundle):
    cand_path, ids_path = cand_paths("test", country)
    cands, ids = pl.read_parquet(cand_path), pl.read_parquet(ids_path)
    df = load_records("test", country)
    assert df["entity_id"].equals(ids["entity_id"]), "record order changed since candidates"
    rec = record_table(df)
    del df
    rule = unseen_rule(bundle, country)
    self_trained = bool(rule and rule.get("self_trained"))
    parts, kept_X, kept_diffs = [], [], []
    for i in range(0, cands.height, CHUNK_PAIRS):
        chunk = cands.slice(i, CHUNK_PAIRS)
        d = name_diffs(chunk, rec)
        f = pl.concat([pair_features(chunk, rec), d.drop(SIDES),
                       te_features(d, bundle["te_stats"], bundle["te_prior"])], how="horizontal")
        out = f.select("s1_rid", "m_rid").with_columns(
            pl.Series("p", predict_mean(bundle["models"], to_matrix(f, bundle["features"]))))
        if rule and rule["model"] is not None:
            out = out.with_columns(pl.Series(
                "p_unseen", rule["model"].predict_proba(to_matrix(f, rule["features"]))[:, 1]))
        if self_trained:
            kept_X.append(to_matrix(f, bundle["features"]))
            kept_diffs.append(d.select(SIDES))
        parts.append(out)
    p = pl.concat(parts)
    eid = ids["entity_id"]
    if self_trained:
        # word scores from this country's own confident pairs, then the full model
        X = np.concatenate(kept_X)
        del kept_X
        fold_of = (eid.gather(p["s1_rid"]).hash(seed=SEED + 1) % N_FOLDS).to_numpy()
        te = pseudo_te_features(pl.concat(kept_diffs), p["p_unseen"].to_numpy(), fold_of, N_FOLDS)
        X[:, [bundle["features"].index(c) for c in te.columns]] = to_matrix(te, te.columns)
        p = p.with_columns(pl.col("p_unseen").alias("p_no_words"),
                           pl.Series("p_unseen", predict_mean(bundle["models"], X)))
        log(f"[{country}] self-trained word scores from its own confident pairs")
    pred = pl.concat([pl.DataFrame({"s1_id": eid.gather(p["s1_rid"]), "match_id": eid.gather(p["m_rid"])}),
                      p.drop("s1_rid", "m_rid")], how="horizontal").with_columns(pl.lit(country).alias("country"))
    log(f"[{country}] scored {pred.height} pairs")
    return pred


def test_s1_ids():
    """(entity_id, country) of every test S1 entity."""
    return pl.concat([
        pl.read_parquet(cand_paths("test", c)[1]).filter(pl.col("src") == "S1")
        .select("entity_id", pl.lit(c).alias("country"))
        for c in list_countries("test")])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reuse-scores", action="store_true",
                    help=f"decide from the cached {PRED_PATH.name} instead of scoring again")
    ap.add_argument("--no-prior-shift", dest="prior_shift", action="store_false",
                    help="skip recalibrating probabilities to each country's EM-estimated "
                         "share of true pairs (on by default)")
    ap.add_argument("--no-unseen-rule", dest="unseen_rule", action="store_false",
                    help="treat countries without training data like the training countries")
    ap.add_argument("--out-dir", default=str(OUTPUT), help="where to write the two TSV files")
    args = ap.parse_args()
    out_dir = Path(args.out_dir) if Path(args.out_dir).is_absolute() else ROOT / args.out_dir

    bundle = joblib.load(MODEL_PATH)
    if args.reuse_scores:
        pred = pl.read_parquet(PRED_PATH)
    else:
        pred = pl.concat([score_country(c, bundle) for c in list_countries("test")],
                         how="diagonal_relaxed")  # p_unseen only for countries without training data
        pred.write_parquet(PRED_PATH)
    s1 = test_s1_ids()

    parts = []
    for country in list_countries("test"):  # one-to-one never crosses countries
        sub = pred.filter(pl.col("country") == country)
        rule = unseen_rule(bundle, country) if args.unseen_rule else None
        use_unseen_model = rule and rule["model"] is not None and "p_unseen" in sub.columns
        p = sub["p_unseen" if use_unseen_model else "p"].to_numpy()
        thr = rule["threshold"] if rule else bundle["threshold"]
        if args.prior_shift:
            p, pi = prior_shift(p, bundle["pi_train"])
            log(f"[{country}] prior shift: estimated true share {pi:.4f} "
                f"(train {bundle['pi_train']:.4f})")
        log(f"[{country}] threshold {thr:.3f}" + (f", no training data: {rule['variant']}" if rule else ""))
        parts.append(decide(sub.select("s1_id", "match_id").with_columns(pl.Series("p", p)), thr))
    matches = pl.concat(parts)

    write_id_lists(pred.select("s1_id", "match_id"), s1["entity_id"],
                   out_dir / "candidate_pairs.tsv", "candidate_entity_ids")
    write_id_lists(matches, s1["entity_id"], out_dir / "matching_results.tsv", "matched_entity_ids")
    per = (s1.join(matches.group_by("s1_id").len("n"), left_on="entity_id", right_on="s1_id",
                   how="left")
             .with_columns(pl.col("n").fill_null(0))
             .group_by("country").agg(pl.len().alias("S1"), (pl.col("n") == 0).mean().alias("empty"),
                                      pl.col("n").mean().alias("matches_per_S1"))
             .join(pred.group_by("country").len("cands"), on="country")
             .with_columns((pl.col("cands") / pl.col("S1")).alias("cands_per_S1"))
             .sort("country"))
    for r in per.iter_rows(named=True):
        log(f"[{r['country']}] {r['S1']} S1: {r['cands_per_S1']:.2f} candidates and "
            f"{r['matches_per_S1']:.2f} matches per S1, {r['empty']:.1%} predicted empty")
    log(f"wrote {out_dir}: {matches.height} matches from {pred.height} candidates "
        f"({'prior shift' if args.prior_shift else 'no prior shift'})")


if __name__ == "__main__":
    main()
