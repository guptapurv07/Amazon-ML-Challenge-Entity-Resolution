"""Export the pairs the pair model is unsure about, with their original text, for
the cross-encoder (fine-tuned and run on a GPU: colab/cross_encoder.ipynb).

The pair model (stage 1) is confident on most pairs. The pairs whose stage-1
probability lies in [LO, HI] ("hard" pairs) get a second opinion from a
multilingual transformer that reads both records' original text, including
Devanagari; stage 2 (stage2.py) then combines both. Written to work/ce/
(upload this folder to Google Drive):

  ce_train.parquet  fine-tuning set: candidates of train S1 entities OUTSIDE the
                    pair-model sample (hash bucket [SAMPLE_PCT, SAMPLE_PCT + CE_PCT)):
                    all hard pairs plus EASY_FRAC of the others, with labels (at most
                    MAX_TRAIN pairs, about 35 min on a T4). Disjoint from the pairs
                    stage 2 is trained and validated on.
  ce_oof.parquet    hard pairs of the pair-model sample, with the out-of-fold stage-1
                    probability and label (stage-2 training data)
  ce_test.parquet   hard test pairs; for a country without training data, hard under
                    either the main model (p) or its own model (p_unseen)
Columns: s1_id, match_id, text_a (S1), text_b (S2/S3), p (stage 1) [, label, country].

After a new pair model or new candidates, `--topup NAME` writes only the hard pairs
that have no cross-encoder score yet (ce_oof_NAME / ce_test_NAME); the notebook then
scores just those (the fine-tuned model is kept on Drive).

Round 2 (`--round2 train`, then `--round2 score`): the round-1 model is fine-tuned
further on ce2_train.parquet, the hard pairs of the train S1 entities used by
neither the pair model nor round 1 (hash bucket [SAMPLE_PCT + CE_PCT, 100)); then
every hard pair of the current pair model is rescored (ce2_oof / ce2_test), see
colab/cross_encoder_round2.ipynb. ce2_train also holds PSEUDO_N confident pairs per
class of each test country WITHOUT training data (self-training): the pair model's
own rule for that country rates them (probability <= PSEUDO_LO -> label 0,
>= PSEUDO_HI -> label 1; never labeled by hand), so the cross-encoder sees that
country's language and formats before it judges the country's hard pairs.
`--no-pseudo` leaves them out.

Run from student_resource/ after train.py and predict.py:
  python src/ce_export.py [--topup v4 | --round2 train | --round2 score]
"""
import machine_limits  # noqa: F401  -- must stay the first import (caps CPU threads)
import argparse

import joblib
import numpy as np
import polars as pl

from candidates import cand_paths, load_records
from data_io import WORK, list_countries, load_gt_pairs, log, read_source
from features import CHUNK_PAIRS, pair_features, record_table
from model import SEED, predict_mean, to_matrix
from predict import unseen_rule
from te import SIDES, name_diffs, te_features
from train import LOCO_COLS, LOCO_PATH, MODEL_PATH, SAMPLE_PCT

CE_DIR = WORK / "ce"
LO, HI = 0.02, 0.98
CE_PCT = 30
EASY_FRAC = 0.03
MAX_TRAIN = 450_000
PSEUDO_LO, PSEUDO_HI = 0.03, 0.97  # confident pairs, as in te.pseudo_te_features
PSEUDO_N = 30_000  # pseudo-labeled pairs per class and country without training data
PSEUDO3_N = 75_000  # round 3: more of them, rated by the current pair model
KEEP3_N = 50_000  # round 3: labeled train pairs from ce2_train, so the model keeps the train countries
LARGE_SEEN_N, LARGE_UNSEEN_N = 330_000, 70_000  # --large: training pairs (labeled / pseudo-labeled)


def hard(p):
    return pl.col(p).is_between(LO, HI)


def finetune_pairs(country, bundle, gt, lo=SAMPLE_PCT, hi=SAMPLE_PCT + CE_PCT):
    """Stage-1 probability and label of every candidate of the S1 entities in hash
    buckets [lo, hi) (the fine-tuning entities)."""
    cand_path, ids_path = cand_paths("train", country)
    ids = pl.read_parquet(ids_path)
    bucket = pl.col("entity_id").hash(seed=SEED) % 100
    s1 = ids.filter((pl.col("src") == "S1") & bucket.is_between(lo, hi - 1))
    cands = pl.read_parquet(cand_path).join(
        s1.select(pl.col("rid").alias("s1_rid")), on="s1_rid", how="semi")
    df = load_records("train", country)
    assert df["entity_id"].equals(ids["entity_id"]), "record order changed since candidates"
    rec = record_table(df)
    del df
    probs = []
    for i in range(0, cands.height, CHUNK_PAIRS):
        chunk = cands.slice(i, CHUNK_PAIRS)
        d = name_diffs(chunk, rec)
        f = pair_features(chunk, rec).hstack(d.drop(SIDES).get_columns())
        f = pl.concat([f, te_features(d, bundle["te_stats"], bundle["te_prior"])], how="horizontal")
        probs.append(predict_mean(bundle["models"], to_matrix(f, bundle["features"])))
    eid = ids["entity_id"]
    truth = gt.join(s1.select(pl.col("entity_id").alias("s1_id")), on="s1_id")
    out = (pl.DataFrame({"s1_id": eid.gather(cands["s1_rid"]), "match_id": eid.gather(cands["m_rid"]),
                         "p": np.concatenate(probs)})
           .join(truth.with_columns(pl.lit(1, pl.Int8).alias("label")), on=["s1_id", "match_id"], how="left")
           .with_columns(pl.col("label").fill_null(0), pl.lit(country).alias("country")))
    log(f"[{country}] fine-tuning set: {s1.height} S1, {out.height} pairs, "
        f"{out.filter(hard('p')).height} hard")
    return out


def add_text(pairs, split):
    """text_a / text_b = 'name | address' of the S1 and the S2/S3 record (original text)."""
    need = pl.concat([pairs.select(pl.col("s1_id").alias("entity_id")),
                      pairs.select(pl.col("match_id").alias("entity_id"))]).unique()
    parts = []
    for source in (1, 2, 3):
        parts.append(read_source(split, source).join(need, on="entity_id", how="semi").select(
            "entity_id", pl.concat_str(pl.col("business_name").fill_null(""), pl.lit(" | "),
                                       pl.col("business_address").fill_null("")).alias("t")))
    text = pl.concat(parts)
    return (pairs.join(text.rename({"entity_id": "s1_id", "t": "text_a"}), on="s1_id")
                 .join(text.rename({"entity_id": "match_id", "t": "text_b"}), on="match_id"))


def pseudo_pairs(bundle, n=PSEUDO_N, seed=SEED + 4):
    """Confident test pairs of the countries without training data, labeled by the
    pair model's rule for them (see the module docstring), or None."""
    pred = pl.read_parquet(WORK / "test_predictions.parquet")
    parts = []
    for country in pred["country"].unique().sort().to_list():
        rule = unseen_rule(bundle, country)
        if rule is None:
            continue
        col = "p_unseen" if rule["model"] is not None and "p_unseen" in pred.columns else "p"
        sub = pred.filter(pl.col("country") == country).select(
            "s1_id", "match_id", pl.col(col).alias("p"), "country").with_columns(
            pl.struct("s1_id", "match_id").hash(seed=seed).alias("k"))
        for label, sure in ((1, pl.col("p") >= PSEUDO_HI), (0, pl.col("p") <= PSEUDO_LO)):
            pool = sub.filter(sure)
            part = pool.sort("k").head(n).drop("k")
            parts.append(part.with_columns(pl.lit(label, pl.Int8).alias("label")))
            log(f"[{country}] pseudo-label {label}: {part.height} of {pool.height} confident pairs ({col})")
    return pl.concat(parts) if parts else None


def scored(name, prefix="ce"):
    """(s1_id, match_id) pairs that already have a cross-encoder score (ce = round 1,
    ce2 = round 2)."""
    files = sorted(CE_DIR.glob(f"{prefix}_scores_{name}*.parquet"))
    if not files:
        return pl.DataFrame(schema={"s1_id": pl.String, "match_id": pl.String})
    return pl.concat([pl.read_parquet(f, columns=["s1_id", "match_id"]) for f in files]).unique()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topup", metavar="NAME",
                    help="only hard pairs without a cross-encoder score, written as "
                         "ce_oof_NAME / ce_test_NAME (no fine-tuning set)")
    ap.add_argument("--round2", choices=["train", "score"],
                    help="round 2: fine-tuning set from the unused train entities, or every "
                         "hard pair of the current pair model")
    ap.add_argument("--no-pseudo", action="store_true",
                    help="round 2 train: no pseudo-labeled pairs of countries without training data")
    ap.add_argument("--ce2", action="store_true",
                    help="with --topup: pairs without a ROUND-2 score (ce2_{oof,test,oof_loco}_NAME), "
                         "for colab/cross_encoder_round2.ipynb")
    ap.add_argument("--round3", choices=["train", "score"],
                    help="round 3, for countries without training data: fine-tuning set (PSEUDO3_N "
                         "confident pairs per class + KEEP3_N labeled pairs of ce2_train), or their "
                         "hard test pairs; see colab/cross_encoder_round3.ipynb")
    ap.add_argument("--large", action="store_true",
                    help="files for the large cross-encoder (kaggle/cross_encoder_large.ipynb): cel_train "
                         "and every hard pair of the current pair model (cel_oof, cel_test, cel_oof_loco)")
    args = ap.parse_args()
    CE_DIR.mkdir(parents=True, exist_ok=True)
    if args.round3:
        bundle = joblib.load(MODEL_PATH)
        if args.round3 == "train":
            keep = pl.read_parquet(CE_DIR / "ce2_train.parquet").filter(
                pl.col("country").is_in(bundle["train_countries"]))
            keep = keep.sort(pl.struct("s1_id", "match_id").hash(seed=SEED + 6)).head(KEEP3_N)
            pseudo = add_text(pseudo_pairs(bundle, PSEUDO3_N, SEED + 5), "test").select(keep.columns)
            ft = pl.concat([keep, pseudo], how="vertical_relaxed")
            ft.write_parquet(CE_DIR / "ce3_train.parquet")
            log(f"ce3_train: {ft.height} pairs ({pseudo.height} pseudo-labeled, {keep.height} labeled), "
                f"positive rate {ft['label'].mean():.3f}")
        else:
            pred = pl.read_parquet(WORK / "test_predictions.parquet")
            parts = []
            for country in pred["country"].unique().sort().to_list():
                rule = unseen_rule(bundle, country)
                if rule is None:
                    continue
                col = "p_unseen" if rule["model"] is not None and "p_unseen" in pred.columns else "p"
                parts.append(pred.filter((pl.col("country") == country) & hard(col)))
            d = pl.concat(parts)
            add_text(d, "test").write_parquet(CE_DIR / "ce3_test.parquet")
            log(f"ce3_test: {d.height} hard pairs of countries without training data")
        return
    if args.large:
        # training set: labeled train pairs of rounds 1 and 2 (S1 entities outside the pair-model
        # sample) + confident pairs of the countries without training data (pseudo-labeled by
        # the pair model); then every hard pair of the current pair model, as for round 2
        tr = pl.concat([pl.read_parquet(CE_DIR / f"{f}.parquet") for f in ("ce_train", "ce2_train", "ce3_train")],
                       how="diagonal_relaxed").unique(["s1_id", "match_id"], keep="first")
        bundle = joblib.load(MODEL_PATH)
        seen = pl.col("country").is_in(bundle["train_countries"])
        key = pl.struct("s1_id", "match_id").hash(seed=SEED + 7)
        tr = pl.concat([tr.filter(seen).sort(key).head(LARGE_SEEN_N),
                        tr.filter(~seen).sort(key).head(LARGE_UNSEEN_N)]).select(
            "s1_id", "match_id", "p", "label", "country", "text_a", "text_b")
        tr.write_parquet(CE_DIR / "cel_train.parquet")
        log(f"cel_train: {tr.height} pairs " + str(dict(tr.group_by("country").len().rows()))
            + f", positive rate {tr['label'].mean():.3f}")
    if args.round2 == "score" or args.large:
        prefix = "cel" if args.large else "ce2"
        for name, path in (("oof", WORK / "oof.parquet"), ("test", WORK / "test_predictions.parquet")):
            d = pl.read_parquet(path)
            is_hard = hard("p")
            if "p_unseen" in d.columns:
                is_hard = is_hard | hard("p_unseen").fill_null(False)
            d = d.filter(is_hard)
            add_text(d, "train" if name == "oof" else "test").write_parquet(CE_DIR / f"{prefix}_{name}.parquet")
            log(f"{prefix}_{name}: {d.height} hard pairs")
            if name == "oof":
                oof_hard = d.select("s1_id", "match_id")
        # pairs that are hard for a model that never saw their country (leave-one-country-out,
        # as for France), for the stage 2 of countries without training data (stage2.py)
        if LOCO_PATH.exists() and LOCO_PATH.stat().st_mtime >= (WORK / "oof.parquet").stat().st_mtime:
            col = f"p_{LOCO_COLS[joblib.load(MODEL_PATH)['unseen']['variant']]}"
            loco = pl.read_parquet(LOCO_PATH).filter(hard(col)).join(
                oof_hard, on=["s1_id", "match_id"], how="anti")
            add_text(loco, "train").write_parquet(CE_DIR / f"{prefix}_oof_loco.parquet")
            log(f"{prefix}_oof_loco: {loco.height} more pairs, hard under leave-one-country-out ({col})")
        return
    if args.round2 == "train":
        bundle, gt = joblib.load(MODEL_PATH), load_gt_pairs()
        ft = pl.concat([finetune_pairs(c, bundle, gt, SAMPLE_PCT + CE_PCT, 100)
                        for c in list_countries("train")])
        easy_kept = pl.struct("s1_id", "match_id").hash(seed=SEED + 2) % 1000 < int(EASY_FRAC * 1000)
        ft = ft.filter(hard("p") | easy_kept)
        if ft.height > MAX_TRAIN:
            rank = pl.struct("s1_id", "match_id").hash(seed=SEED + 3).rank("ordinal")
            ft = ft.filter(rank <= MAX_TRAIN)
        ft = add_text(ft, "train")
        pseudo = None if args.no_pseudo else pseudo_pairs(bundle)
        if pseudo is not None:
            ft = pl.concat([ft, add_text(pseudo, "test").select(ft.columns)], how="vertical_relaxed")
        ft.write_parquet(CE_DIR / "ce2_train.parquet")
        log(f"ce2_train: {ft.height} pairs ({0 if pseudo is None else pseudo.height} pseudo-labeled), "
            f"positive rate {ft['label'].mean():.3f}")
        return
    if args.topup:
        prefix = "ce2" if args.ce2 else "ce"
        for name, path in (("oof", WORK / "oof.parquet"), ("test", WORK / "test_predictions.parquet")):
            d = pl.read_parquet(path)
            is_hard = hard("p")
            if "p_unseen" in d.columns:
                is_hard = is_hard | hard("p_unseen").fill_null(False)
            d = d.filter(is_hard).join(scored(name, prefix), on=["s1_id", "match_id"], how="anti")
            add_text(d, "train" if name == "oof" else "test").write_parquet(
                CE_DIR / f"{prefix}_{name}_{args.topup}.parquet")
            log(f"{prefix}_{name}_{args.topup}: {d.height} hard pairs without a score")
            if name == "oof":
                oof_new = d.select("s1_id", "match_id")
        if args.ce2 and LOCO_PATH.exists() and LOCO_PATH.stat().st_mtime >= (WORK / "oof.parquet").stat().st_mtime:
            col = f"p_{LOCO_COLS[joblib.load(MODEL_PATH)['unseen']['variant']]}"
            loco = (pl.read_parquet(LOCO_PATH).filter(hard(col))
                      .join(scored("oof", prefix), on=["s1_id", "match_id"], how="anti")
                      .join(oof_new, on=["s1_id", "match_id"], how="anti"))
            add_text(loco, "train").write_parquet(CE_DIR / f"ce2_oof_loco_{args.topup}.parquet")
            log(f"ce2_oof_loco_{args.topup}: {loco.height} leave-one-country-out hard pairs without a score")
        return
    bundle = joblib.load(MODEL_PATH)
    gt = load_gt_pairs()

    ft = pl.concat([finetune_pairs(c, bundle, gt) for c in list_countries("train")])
    easy_kept = pl.struct("s1_id", "match_id").hash(seed=SEED + 2) % 1000 < int(EASY_FRAC * 1000)
    ft = ft.filter(hard("p") | easy_kept)
    if ft.height > MAX_TRAIN:
        rank = pl.struct("s1_id", "match_id").hash(seed=SEED + 3).rank("ordinal")
        ft = ft.filter(rank <= MAX_TRAIN)
    add_text(ft, "train").write_parquet(CE_DIR / "ce_train.parquet")
    log(f"ce_train: {ft.height} pairs, positive rate {ft['label'].mean():.3f}")

    oof = pl.read_parquet(WORK / "oof.parquet").filter(hard("p"))
    add_text(oof, "train").write_parquet(CE_DIR / "ce_oof.parquet")
    log(f"ce_oof: {oof.height} hard pairs of the pair-model sample")

    test = pl.read_parquet(WORK / "test_predictions.parquet")
    is_hard = hard("p")
    if "p_unseen" in test.columns:
        is_hard = is_hard | hard("p_unseen").fill_null(False)
    test = test.filter(is_hard)
    add_text(test, "test").write_parquet(CE_DIR / "ce_test.parquet")
    log(f"ce_test: {test.height} hard test pairs " + str(dict(test.group_by("country").len().rows())))


if __name__ == "__main__":
    main()
