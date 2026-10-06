"""Measure blocking quality on train: pair completeness (recall ceiling),
candidates per S1, per-family contribution, and the oracle macro F0.5 (the
score a perfect matcher would get on these candidates).

Run from student_resource/:
  python src/blocking_eval.py                      # every country in train
  python src/blocking_eval.py --country India      # one country
  python src/blocking_eval.py --country India --frac 0.02   # smoke test only
--frac keeps a random fraction of S1 plus all S2/S3, so blocks are smaller
than in the full data: use it to check the code runs, not for numbers.
Raw pairs are kept in work/blk/train/<country>/ for the later pipeline steps.
"""
import machine_limits  # noqa: F401  -- must stay the first import (caps CPU threads)
import argparse
import time

import numpy as np
import polars as pl

from blocking import (FAMILIES, N_CHUNKS, blk_dir, chunk_path, encode_pair, family_bit,
                      generate_candidates)
from data_io import list_countries, load_gt_pairs, load_norm, log

BLOCKING_COLS = ["entity_id", "src", "had_nonascii", "name_core", "addr_norm", "addr_nums"]


def load_country(country, frac=1.0):
    df = load_norm("train", country, columns=BLOCKING_COLS)
    if frac < 1:
        s1 = df.filter(pl.col("src") == "S1").sample(fraction=frac, seed=0)
        df = pl.concat([s1, df.filter(pl.col("src") != "S1")])
    return df.with_row_index("rid")


def lookup(keys, bits, queries):
    """For sorted u64 `keys`, return (found, bits) of each query."""
    if len(keys) == 0:
        return np.zeros(len(queries), bool), np.zeros(len(queries), np.uint8)
    pos = np.minimum(np.searchsorted(keys, queries), len(keys) - 1)
    found = keys[pos] == queries
    return found, np.where(found, bits[pos], 0).astype(np.uint16)


def evaluate_country(df, gt, country, out_dir):
    t = time.time()
    n_pairs = generate_candidates(df, out_dir)
    s1 = df.filter(pl.col("src") == "S1").select(
        pl.col("rid").alias("s1_rid"), pl.col("entity_id").alias("s1_id"))
    m = df.filter(pl.col("src") != "S1").select(
        pl.col("rid").alias("m_rid"), pl.col("entity_id").alias("match_id"),
        (pl.col("addr_norm") == "").alias("m_addr_empty"),
        pl.col("had_nonascii").alias("m_nonascii"))
    truth = gt.join(s1, on="s1_id").join(m, on="match_id").with_columns(
        encode_pair(pl.col("s1_rid"), pl.col("m_rid")).alias("pair"))
    hits, fam_pairs = [], dict.fromkeys(FAMILIES, 0)
    for c in range(N_CHUNKS):
        cp = pl.read_parquet(chunk_path(out_dir, c))
        keys, bits = cp["pair"].to_numpy(), cp["bits"].to_numpy()
        for fam in FAMILIES:
            fam_pairs[fam] += int(((bits & family_bit(fam)) > 0).sum())
        tc = truth.filter(pl.col("s1_rid") % N_CHUNKS == c)
        found, b = lookup(keys, bits, tc["pair"].to_numpy())
        hits.append(tc.with_columns(pl.Series("found", found), pl.Series("bits", b)))
        del cp, keys, bits
    hit = pl.concat(hits)
    n_true, n_s1, n_m = truth.height, s1.height, m.height

    per_s1 = hit.group_by("s1_rid").agg(pl.len().alias("n_true"), pl.col("found").sum().alias("tp"))
    r = pl.col("tp") / pl.col("n_true")
    f_oracle = per_s1.select(
        pl.when(pl.col("tp") == 0).then(0.0).otherwise(1.25 * r / (0.25 + r)).sum()
    ).item()
    oracle = (f_oracle + (n_s1 - per_s1.height)) / n_s1  # singletons score 1.0

    log(f"[{country}] S1={n_s1} S2/S3={n_m} true pairs={n_true} "
        f"candidates={n_pairs} ({n_pairs / n_s1:.1f}/S1) "
        f"PC={hit['found'].mean():.4f} RR={1 - n_pairs / (n_s1 * n_m):.6f} "
        f"oracle_F0.5={oracle:.4f} ({time.time() - t:.0f}s)")
    for label, cond in [("addr empty", pl.col("m_addr_empty")),
                        ("non-ASCII name", pl.col("m_nonascii")),
                        ("other", ~pl.col("m_addr_empty") & ~pl.col("m_nonascii"))]:
        sub = hit.filter(cond)
        log(f"   PC on {label:14s}: {sub['found'].mean():.4f} ({sub.height / n_true:.1%} of true pairs)")
    for fam in FAMILIES:
        b = pl.lit(family_bit(fam), dtype=pl.UInt16)
        any_ = hit.filter((pl.col("bits") & b) > 0).height
        only = hit.filter(pl.col("bits") == b).height
        log(f"   {fam:5s} recall={any_ / n_true:.4f} unique={only / n_true:.4f} "
            f"pairs/S1={fam_pairs[fam] / n_s1:.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--country", help="evaluate only this country (default: all in train)")
    ap.add_argument("--frac", type=float, default=1.0,
                    help="fraction of S1 to keep (smoke tests only; distorts block sizes)")
    args = ap.parse_args()
    countries = [args.country] if args.country else list_countries("train")
    gt = load_gt_pairs()
    for country in countries:
        log(f"loading {country}")
        df = load_country(country, args.frac)
        out_dir = blk_dir("train", country) if args.frac == 1.0 else blk_dir("smoke", country)
        evaluate_country(df, gt, country, out_dir)
        del df


if __name__ == "__main__":
    main()
