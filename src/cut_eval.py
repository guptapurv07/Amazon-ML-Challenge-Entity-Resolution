"""Choose the candidate cut on train: candidates per S1 vs recall ceiling.

For every train country: take the cheap-scored raw blocking pairs (written by
candidates.py, or scored here if missing), rank them per S2/S3 record, and
evaluate cut rules under several cheap-score variants:
  cand/S1  average candidates per S1 entity (smaller ranks higher in the final evaluation)
  PC       pair completeness = share of true pairs kept (recall ceiling)
  oracle   macro F0.5 a perfect matcher would reach on the kept candidates
The report is also written to work/cut_eval_report.txt.

Run from student_resource/:  python src/cut_eval.py [--country India]
"""
import machine_limits  # noqa: F401  -- must stay the first import (caps CPU threads)
import argparse

import numpy as np
import polars as pl

from blocking import N_CHUNKS, chunk_path
from candidates import base_expr, cut_mask, group_ranks, load_records, load_scored, score_raw, scored_dir
from data_io import WORK, list_countries, load_gt_pairs, log
from features import record_table

SCORES = {
    "v2 score": pl.max_horizontal("name_tsr", "cat_ratio").cast(pl.Float32)
                + 0.5 * pl.col("addr_tsr").cast(pl.Float32),
    "+acronym": base_expr(w_num=0.0),
    "+acronym +0.25 number": base_expr(w_num=0.25),
    "+acronym +0.5 number": base_expr(w_num=0.5),
}
RULES = [
    ("record best S1 only", dict(m_max=1, delta=0)),
    ("best + 2nd within 10", dict(m_max=2, delta=10)),
    ("best + 2nd,3rd within 10", dict(m_max=3, delta=10)),
    ("best + 2nd,3rd within 15", dict(m_max=3, delta=15)),
]


def truth_rows(raw, country, gt):
    """Per true pair of the country: its S1 rid and its row in `raw` (-1 if
    blocking never produced it). Also returns the S1 rids and the rid count."""
    ids = load_records("train", country).select("rid", "entity_id", "src")
    s1_ids = ids.filter(pl.col("src") == "S1")
    truth = (gt.join(s1_ids.select(pl.col("rid").alias("s1_rid"), pl.col("entity_id").alias("s1_id")),
                     on="s1_id")
               .join(ids.select(pl.col("rid").alias("m_rid"), pl.col("entity_id").alias("match_id")),
                     on="match_id")
               .select("s1_rid", "m_rid"))
    found = truth.join(raw.select("s1_rid", "m_rid").with_row_index("row"),
                       on=["s1_rid", "m_rid"], how="left")
    return (found["s1_rid"].to_numpy(), found["row"].fill_null(-1).cast(pl.Int64).to_numpy(),
            s1_ids["rid"].to_numpy(), ids.height)


def oracle(keep, t_s1, t_row, s1_rids, n_rid):
    """(pair completeness, oracle macro F0.5) of the kept pairs."""
    kept = t_row >= 0
    kept[kept] = keep[t_row[kept]]
    n_true = np.bincount(t_s1, minlength=n_rid)[s1_rids]
    tp = np.bincount(t_s1[kept], minlength=n_rid)[s1_rids]
    r = np.divide(tp, n_true, out=np.zeros(len(tp)), where=n_true > 0)
    f = np.where(n_true == 0, 1.0, np.where(tp == 0, 0.0, 1.25 * r / (0.25 + np.maximum(r, 1e-12))))
    return kept.mean(), f.mean()


def evaluate(country, gt, out):
    if not chunk_path(scored_dir("train", country), N_CHUNKS - 1).exists():
        rec = record_table(load_records("train", country))
        log(f"[{country}] cheap-scored {score_raw('train', country, rec)} raw pairs")
        del rec
    raw = load_scored("train", country)
    t_s1, t_row, s1_rids, n_rid = truth_rows(raw, country, gt)
    n_s1 = len(s1_rids)
    s1, m = raw["s1_rid"].to_numpy(), raw["m_rid"].to_numpy()
    pc, f = oracle(np.ones(raw.height, bool), t_s1, t_row, s1_rids, n_rid)
    out(f"\n===== {country}: {n_s1} S1, {len(t_s1)} true pairs")
    out(f"{'score / rule':58s} {'cand/S1':>8s} {'PC':>7s} {'oracle':>7s}")
    out(f"{'raw blocking (no cut)':58s} {raw.height / n_s1:8.2f} {pc:7.4f} {f:7.4f}")
    for score_name, expr in SCORES.items():
        base = raw.select(expr).to_series().to_numpy()
        m_rank, m_gid, _, m_top1, _ = group_ranks(m, base)
        m_top1_row = m_top1[m_gid]
        del m_gid
        for label, rule in RULES:
            keep = cut_mask(base, m_rank, m_top1_row, None, s1, **rule)
            pc, f = oracle(keep, t_s1, t_row, s1_rids, n_rid)
            out(f"{score_name + ': ' + label:58s} {keep.sum() / n_s1:8.2f} {pc:7.4f} {f:7.4f}")
        del base, m_rank, m_top1_row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--country", help="evaluate only this country (default: all in train)")
    args = ap.parse_args()
    gt = load_gt_pairs()
    with open(WORK / "cut_eval_report.txt", "w") as fh:
        def out(line):
            log(line)
            fh.write(line + "\n")
            fh.flush()
        for country in [args.country] if args.country else list_countries("train"):
            evaluate(country, gt, out)


if __name__ == "__main__":
    main()
