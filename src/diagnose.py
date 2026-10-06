"""Where does the score go? Run on the CV sample of train S1 entities.

  1. Oracle macro F0.5 after each stage (a perfect matcher on that pair set):
     raw blocking -> trimmed candidates -> actual model (OOF + decision rule)
  2. Loss decomposition of the model: sum of (1 - F0.5) per S1 by cause
  3. Raw-text samples of each error type
Report: work/diagnose_report.txt. Needs work/blk, work/cand, work/oof*.parquet
and work/model.joblib (i.e. candidates.py and train.py have run).

Run from student_resource/:  python src/diagnose.py
"""
import machine_limits  # noqa: F401  -- must stay the first import (caps CPU threads)
import joblib
import numpy as np
import polars as pl

from blocking import N_CHUNKS, blk_dir, chunk_path, encode_pair
from candidates import cand_paths
from data_io import WORK, list_countries, load_gt_pairs, source_path
from decide import decide
from metrics import per_entity_f05
from train import MODEL_PATH

pl.Config.set_fmt_str_lengths(70)
pl.Config.set_tbl_width_chars(250)
pl.Config.set_tbl_rows(30)
N_SAMPLES = 12
LINES = []


def out(*args):
    text = " ".join(str(a) for a in args)
    print(text, flush=True)
    LINES.append(text)


def lookup(keys, queries):
    if len(keys) == 0:
        return np.zeros(len(queries), bool)
    pos = np.minimum(np.searchsorted(keys, queries), len(keys) - 1)
    return keys[pos] == queries


def oracle(truth_found, n_s1):
    """Macro F0.5 of a perfect matcher that can only pick pairs marked found."""
    per = truth_found.group_by("s1_id").agg(pl.len().alias("n"), pl.col("found").sum().alias("tp"))
    r = pl.col("tp") / pl.col("n")
    f_sum = per.select(pl.when(pl.col("tp") == 0).then(0.0)
                       .otherwise(1.25 * r / (0.25 + r)).sum()).item()
    return (f_sum + n_s1 - per.height) / n_s1  # S1 without true matches: predict empty -> 1.0


def stage_membership(country, truth):
    """Mark each true pair as present in the raw blocking output / trimmed candidates."""
    ids = pl.read_parquet(cand_paths("train", country)[1]).select("rid", "entity_id")
    t = (truth.join(ids.rename({"rid": "s1_rid", "entity_id": "s1_id"}), on="s1_id")
              .join(ids.rename({"rid": "m_rid", "entity_id": "match_id"}), on="match_id")
              .with_columns(encode_pair(pl.col("s1_rid"), pl.col("m_rid")).alias("pair")))
    raw_found = np.zeros(t.height, bool)
    for c in range(N_CHUNKS):
        keys = pl.read_parquet(chunk_path(blk_dir("train", country), c), columns=["pair"])["pair"].to_numpy()
        raw_found |= lookup(keys, t["pair"].to_numpy())
    cand = pl.read_parquet(cand_paths("train", country)[0], columns=["s1_rid", "m_rid"])
    keys = np.sort(cand.select(encode_pair(pl.col("s1_rid"), pl.col("m_rid")))
                   .to_series().to_numpy())
    trim_found = lookup(keys, t["pair"].to_numpy())
    return t.select("s1_id", "match_id").with_columns(
        pl.Series("in_raw", raw_found), pl.Series("in_cand", trim_found))


def raw_text(ids):
    ids = list(set(ids))
    frames = [pl.scan_csv(source_path("train", s), separator="\t", infer_schema=False,
                          quote_char='"').filter(pl.col("entity_id").is_in(ids)).collect()
              for s in (1, 2, 3)]
    return pl.concat(frames).select("entity_id", "business_name", "business_address")


def show(title, pairs, text, extra_cols=()):
    if pairs.height == 0:
        return
    out(f"\n--- {title} ({pairs.height} shown)")
    t1 = text.rename({"entity_id": "s1_id", "business_name": "s1_name", "business_address": "s1_addr"})
    t2 = text.rename({"entity_id": "match_id", "business_name": "m_name", "business_address": "m_addr"})
    out(pairs.join(t1, on="s1_id", how="left").join(t2, on="match_id", how="left")
        .select(*extra_cols, "s1_name", "m_name", "s1_addr", "m_addr"))


def main():
    bundle = joblib.load(MODEL_PATH)
    t = bundle["threshold"]
    oof = pl.read_parquet(WORK / "oof.parquet")
    s1_all = pl.read_parquet(WORK / "oof_s1.parquet")
    gt = load_gt_pairs()
    samples = []
    for country in list_countries("train"):
        s1 = s1_all.filter(pl.col("country") == country)
        truth = gt.join(s1.select("s1_id"), on="s1_id")
        mem = stage_membership(country, truth)
        n = s1.height
        pred = oof.filter(pl.col("country") == country).select("s1_id", "match_id", "p")
        chosen = decide(pred, t)
        per = per_entity_f05(chosen, truth, s1["s1_id"])
        out(f"\n===== {country}: {n} sampled S1, {truth.height} true pairs")
        out(f"  pair recall      raw blocking {mem['in_raw'].mean():.4f} | trimmed {mem['in_cand'].mean():.4f}")
        out(f"  oracle F0.5      raw blocking {oracle(mem.rename({'in_raw': 'found'}), n):.4f} | "
            f"trimmed {oracle(mem.rename({'in_cand': 'found'}), n):.4f} | model {per['f05'].mean():.4f}")

        # loss decomposition
        cand_true = mem.group_by("s1_id").agg(pl.col("in_cand").sum().alias("n_in_cand"))
        per = (per.join(cand_true, on="s1_id", how="left").with_columns(pl.col("n_in_cand").fill_null(0))
               .with_columns(
                   (pl.col("n_true") - pl.col("n_in_cand")).alias("miss_cand"),
                   (pl.col("n_in_cand") - pl.col("tp")).alias("miss_model"),
                   (pl.col("n_pred") - pl.col("tp")).alias("fp"))
               .with_columns(
                   pl.when(pl.col("f05") >= 1.0).then(pl.lit("perfect"))
                   .when((pl.col("fp") > 0) & (pl.col("n_true") == 0)).then(pl.lit("FP on a singleton"))
                   .when(pl.col("fp") > 0).then(pl.lit("wrong match (FP)"))
                   .when((pl.col("miss_cand") > 0) & (pl.col("miss_model") == 0)).then(pl.lit("miss: not a candidate"))
                   .when((pl.col("miss_model") > 0) & (pl.col("miss_cand") == 0)).then(pl.lit("miss: model rejected"))
                   .otherwise(pl.lit("miss: both")).alias("cause")))
        loss = (per.group_by("cause").agg(pl.len().alias("n_s1"), (1 - pl.col("f05")).sum().alias("loss"))
                .with_columns((pl.col("loss") / n).round(4).alias("F0.5 points lost"))
                .sort("loss", descending=True))
        out("  loss by cause (F0.5 points lost = share of the macro average):")
        out(loss)

        # samples of each error type
        rng = 0
        blk_miss = mem.filter(~pl.col("in_raw")).sample(min(N_SAMPLES, mem.filter(~pl.col("in_raw")).height), seed=rng)
        trim_miss = mem.filter(pl.col("in_raw") & ~pl.col("in_cand"))
        trim_miss = trim_miss.sample(min(N_SAMPLES // 2, trim_miss.height), seed=rng)
        truth_set = truth.with_columns(pl.lit(True).alias("is_true"))
        fps = chosen.join(truth_set, on=["s1_id", "match_id"], how="left").filter(pl.col("is_true").is_null())
        # which S1 does the wrongly matched record really belong to (if any)?
        owner = gt.rename({"s1_id": "true_s1"})
        fps = fps.join(owner, on="match_id", how="left")
        fps = fps.sample(min(N_SAMPLES, fps.height), seed=rng)
        rejected = (pred.join(truth_set, on=["s1_id", "match_id"]).filter(pl.col("p") < t))
        rejected = rejected.sample(min(N_SAMPLES // 2, rejected.height), seed=rng)
        samples.append((country, blk_miss, trim_miss, fps, rejected))

    ids = []
    for _, a, b, c, d in samples:
        for df in (a, b, c, d):
            ids += df["s1_id"].to_list() + df["match_id"].to_list()
        ids += [x for x in c["true_s1"].to_list() if x]
    text = raw_text(ids)
    for country, blk_miss, trim_miss, fps, rejected in samples:
        out(f"\n\n########## {country} samples")
        show("MISSED BY BLOCKING (true pair never a candidate)", blk_miss, text)
        show("LOST AT TRIM (found by blocking, cut by top-K)", trim_miss, text)
        show("WRONG MATCHES (predicted, not true); true_s1 = real owner of the record", fps, text,
             extra_cols=("p", "true_s1"))
        show("REJECTED BY MODEL (true candidate below threshold)", rejected, text, extra_cols=("p",))
    (WORK / "diagnose_report.txt").write_text("\n".join(LINES))


if __name__ == "__main__":
    main()
