"""Final candidate set: blocking -> cheap scores -> one-to-one cut.

For one (split, country):
  1. blocking.generate_candidates  raw key-based pairs (~45 per S1), per S1 chunk
  2. features.cheap_scores         cheap name / address similarity of EVERY raw pair,
                                   cached in work/scored/<split>/<country>/
  3. group_ranks                   rank of each pair among the raw candidates of its
                                   S2/S3 record, and among those of its S1
  4. cut_mask                      one-to-one cut: every S2/S3 record belongs to at
                                   most one S1 (verified on the train labels), so a
                                   record is kept only under its best S1, plus close
                                   runners-up, above a minimum cheap score
  5. competition features for the kept pairs (count / rank / gap / margin among
     the raw candidates of the S1 and of the record)
The kept pairs are exactly what the matching model scores, i.e. what goes into
output/candidate_pairs.tsv. The cut ranks a record's S1 candidates with a small
model learned on train (cut_model.py -> work/cut_model.joblib: cheap similarities
+ blocking families, cross-fitted on train); without that file it falls back to
the hand-weighted BASE with the rule chosen by cut_eval.py.

Files (rids index the country's record table; the ids file maps them back):
  work/cand/<split>/<country>.parquet
  work/cand/<split>/<country>_ids.parquet

Run from student_resource/:
  python src/candidates.py --split train --country India
  python src/candidates.py            # every split and country, one process each
"""
import machine_limits  # noqa: F401  -- must stay the first import (caps CPU threads)
import argparse
import shutil
import subprocess
import sys

import joblib
import numpy as np
import polars as pl

from blocking import FAMILIES, N_CHUNKS, blk_dir, chunk_path, decode_pairs, family_bit, generate_candidates
from data_io import WORK, country_slug, list_countries, load_norm, log
from features import RECORD_COLS, cheap_scores, record_table

SPLITS = ("train", "test")
CHEAP_COLS = ["name_tsr", "cat_ratio", "addr_tsr", "num_tsr", "acr"]


def base_expr(w_num=0.25):
    """Cheap score used for all ranking: the best name score (token set, joined
    name, acronym) + half the address score + w_num x the house-number score."""
    return (pl.max_horizontal("name_tsr", "cat_ratio", "acr").cast(pl.Float32)
            + 0.5 * pl.col("addr_tsr").cast(pl.Float32)
            + w_num * pl.col("num_tsr").cast(pl.Float32))


BASE = base_expr()
# One-to-one cut, chosen with cut_eval.py (see experiments.md):
#   m_max   at most this many S1 per S2/S3 record
#   delta   runners-up must be within this cheap-score distance of the record's best
#   floor   minimum cheap score
#   s1_topk also keep each S1's top-k pairs (0 = off)
#   s1_max  at most this many kept records per S1 (0 = no cap)
CUT = {"m_max": 3, "delta": 10.0, "floor": 0.0, "s1_topk": 0, "s1_max": 0}
CUT_MODEL = WORK / "cut_model.joblib"
SCORE_ROWS = 5_000_000


def cut_matrix(raw, families=FAMILIES):
    """Inputs of the learned cut score: cheap similarities + blocking-family flags."""
    cols = [raw[c].to_numpy().astype(np.float32) for c in CHEAP_COLS]
    bits = raw["bits"].to_numpy()
    cols += [((bits & family_bit(f)) > 0).astype(np.float32) for f in families]
    return np.column_stack(cols)


def cut_score(raw, split):
    """(score, rule) for the cut: the learned logit when work/cut_model.joblib exists,
    else the hand-weighted BASE. On train a pair is scored by the model fitted on the
    other half of the S1 entities (cross-fitted); elsewhere by the mean of both."""
    if not CUT_MODEL.exists():
        return raw.select(BASE).to_series().to_numpy(), CUT
    cm = joblib.load(CUT_MODEL)
    half = (raw["s1_rid"].to_numpy() % 2).astype(np.int8)
    score = np.empty(raw.height, np.float32)
    for start in range(0, raw.height, SCORE_ROWS):
        sl = slice(start, start + SCORE_ROWS)
        X = cut_matrix(raw[sl], [f[4:] for f in cm["features"] if f.startswith("blk_")])
        if split == "train":
            for h in (0, 1):
                sel = half[sl] == h
                if sel.any():  # pairs are stored by S1 chunk, so a slice may hold one half only
                    score[sl][sel] = cm["models"][h].decision_function(X[sel])
        else:
            score[sl] = np.mean([m.decision_function(X) for m in cm["models"]], axis=0)
    return score, {**CUT, **cm["rule"]}


def cand_paths(split, country):
    d = WORK / "cand" / split
    slug = country_slug(country)
    return d / f"{slug}.parquet", d / f"{slug}_ids.parquet"


def scored_dir(split, country):
    return WORK / "scored" / split / country_slug(country)


def load_records(split, country):
    """Country records with rid = row position."""
    return load_norm(split, country, columns=RECORD_COLS).with_row_index("rid")


def score_raw(split, country, rec):
    """Cheap-score every raw blocking pair; written per S1 chunk."""
    src, dst = blk_dir(split, country), scored_dir(split, country)
    shutil.rmtree(dst, ignore_errors=True)
    dst.mkdir(parents=True)
    total = 0
    for c in range(N_CHUNKS):
        p = cheap_scores(decode_pairs(pl.read_parquet(chunk_path(src, c))), rec)
        p.write_parquet(chunk_path(dst, c))
        total += p.height
    return total


def load_scored(split, country):
    d = scored_dir(split, country)
    return pl.concat([pl.read_parquet(chunk_path(d, c)) for c in range(N_CHUNKS)])


def group_ranks(key, score):
    """Rank pairs within groups of `key` by descending `score` (1 = best; ties
    keep input order, so results are deterministic).

    Returns (rank, gid, size, top1, top2): per pair its rank and group index;
    per group its size, best and second-best score (NaN if the group has one pair).
    """
    n = len(key)
    if n == 0:
        z = np.zeros(0, np.uint32)
        return z, z.astype(np.int32), z, z.astype(np.float32), z.astype(np.float32)
    order = np.lexsort((-score, key))
    ks = key[order]
    starts = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1]]).astype(np.int32)
    del ks
    size = np.diff(np.r_[starts, n]).astype(np.uint32)
    gid_sorted = np.repeat(np.arange(len(starts), dtype=np.int32), size)
    rank_sorted = (np.arange(n, dtype=np.int32) - starts[gid_sorted] + 1).astype(np.uint32)
    ss = score[order]
    top1 = ss[starts].astype(np.float32)
    top2 = np.where(size > 1, ss[np.minimum(starts + 1, n - 1)], np.nan).astype(np.float32)
    del ss
    rank = np.empty(n, np.uint32)
    rank[order] = rank_sorted
    gid = np.empty(n, np.int32)
    gid[order] = gid_sorted
    return rank, gid, size, top1, top2


def cut_mask(base, m_rank, m_top1_row, s1_rank, s1, m_max, delta, floor=0.0, s1_topk=0, s1_max=0):
    """Boolean mask of the pairs kept by the one-to-one cut (see CUT)."""
    keep = (m_rank == 1) | ((m_rank <= m_max) & (m_top1_row - base <= delta))
    if s1_topk:
        keep |= s1_rank <= s1_topk
    if floor:
        keep &= base >= floor
    if s1_max:
        idx = np.flatnonzero(keep)
        r = group_ranks(s1[idx], base[idx])[0]
        keep[idx[r > s1_max]] = False
    return keep


def competition(base, ranks, idx, suffix):
    """count / rank / gap to the best / margin over the best OTHER candidate,
    among the raw candidates of the group (100 when there is no other)."""
    rank, gid, size, top1, top2 = ranks
    g, b, r = gid[idx], base[idx], rank[idx]
    other = np.where(r == 1, top2[g], top1[g])
    return {
        f"n_cand_{suffix}": size[g],
        f"rank_{suffix}": r,
        f"gap_{suffix}": (b - top1[g]).astype(np.float32),
        f"margin_{suffix}": np.where(np.isnan(other), 100.0, b - other).astype(np.float32),
    }


def cut_candidates(raw, split):
    """Apply the one-to-one cut to scored raw pairs and add competition features."""
    base, cut = cut_score(raw, split)
    s1, m = raw["s1_rid"].to_numpy(), raw["m_rid"].to_numpy()
    m_ranks = group_ranks(m, base)
    s1_ranks = group_ranks(s1, base)
    keep = cut_mask(base, m_ranks[0], m_ranks[3][m_ranks[1]], s1_ranks[0], s1, **cut)
    idx = np.flatnonzero(keep)
    del keep
    comp = {**competition(base, s1_ranks, idx, "s1"), **competition(base, m_ranks, idx, "m")}
    return raw[idx].with_columns(
        pl.col(CHEAP_COLS).cast(pl.Float32),
        pl.Series("base", base[idx]),
        *[pl.Series(k, v) for k, v in comp.items()],
    )


def build_candidates(split, country, reuse_blocking=False, reuse_scores=False):
    """Build and save the final candidate set of one country."""
    df = load_records(split, country)
    n_s1 = df.filter(pl.col("src") == "S1").height
    blk = blk_dir(split, country)
    reuse_blocking = reuse_blocking and chunk_path(blk, N_CHUNKS - 1).exists()
    if reuse_blocking:
        log(f"[{split}/{country}] reusing blocking output")
    else:
        n_raw = generate_candidates(df, blk)
        log(f"[{split}/{country}] blocking: {n_raw} raw pairs ({n_raw / n_s1:.1f}/S1)")
    ids_path = cand_paths(split, country)[1]
    ids_path.parent.mkdir(parents=True, exist_ok=True)
    df.select("rid", "entity_id", "src").write_parquet(ids_path)
    if reuse_blocking and reuse_scores and chunk_path(scored_dir(split, country), N_CHUNKS - 1).exists():
        log(f"[{split}/{country}] reusing cheap scores")
    else:
        rec = record_table(df)
        n_raw = score_raw(split, country, rec)
        log(f"[{split}/{country}] cheap-scored {n_raw} raw pairs")
        del rec
    del df
    cands = cut_candidates(load_scored(split, country), split)
    cands.write_parquet(cand_paths(split, country)[0])
    log(f"[{split}/{country}] kept {cands.height} candidates ({cands.height / n_s1:.2f}/S1)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=SPLITS)
    ap.add_argument("--country")
    ap.add_argument("--reuse-blocking", action="store_true")
    ap.add_argument("--reuse-scores", action="store_true")
    args = ap.parse_args()
    flags = [f for f, on in (("--reuse-blocking", args.reuse_blocking),
                             ("--reuse-scores", args.reuse_scores)) if on]
    if args.split and args.country:
        build_candidates(args.split, args.country, args.reuse_blocking, args.reuse_scores)
        return
    # one subprocess per (split, country) so memory is fully released in between
    for split in [args.split] if args.split else SPLITS:
        for country in list_countries(split):
            subprocess.run([sys.executable, __file__, "--split", split, "--country", country,
                            *flags], check=True)


if __name__ == "__main__":
    main()
