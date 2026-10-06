"""Countries without training data: settle the pairs where two versions of the teammate
pipeline disagree, with this pipeline's probability as the third opinion.

Starts from the OLDER teammate version's matches (best on the public leaderboard) and
  * drops a pair it matched when the NEWER version dropped it and this pipeline's final
    probability is below --drop-below (two of three confidently say no);
  * adds a pair the newer version matched and the older did not when this pipeline's
    probability is at least --add-above (two of three say yes) and the S2/S3 record is not
    already matched to another S1 entity.
Pairs outside this pipeline's candidates have no probability and are never changed.
Writes a matching_results-style TSV for `team_merge.py --mate-unseen-file`. Only countries
without training data (test countries absent from train) are written.

Run from student_resource/:
  python src/france_vote.py --old <older teammate TSV> --new <newer teammate TSV> \
      --ours work/sub_final --out work/inputs/unseen_vote.tsv
"""
import machine_limits  # noqa: F401  -- must stay the first import (caps CPU threads)
import argparse
from pathlib import Path

import polars as pl

from data_io import ROOT, list_countries, log, write_id_lists
from predict import test_s1_ids
from team_merge import id_pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True, help="older teammate matching_results.tsv (the base)")
    ap.add_argument("--new", required=True, help="newer teammate matching_results.tsv")
    ap.add_argument("--ours", required=True, help="stage2.py --save-test-probs output folder")
    ap.add_argument("--drop-below", type=float, default=0.1)
    ap.add_argument("--add-above", type=float, default=0.725)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    s1 = test_s1_ids()
    unseen = [c for c in list_countries("test") if c not in list_countries("train")]
    s1u = s1.filter(pl.col("country").is_in(unseen)).select(pl.col("entity_id").alias("s1_id"))
    old = id_pairs(args.old, "matched_entity_ids").join(s1u, on="s1_id", how="semi")
    new = id_pairs(args.new, "matched_entity_ids").join(s1u, on="s1_id", how="semi")
    q = pl.read_parquet(Path(args.ours) / "test_probs.parquet").join(s1u, on="s1_id", how="semi").select(
        "s1_id", "match_id", "q")
    key = ["s1_id", "match_id"]

    old_only = old.join(new, on=key, how="anti").join(q, on=key, how="left")
    drop = old_only.filter(pl.col("q") < args.drop_below).select(key)
    kept = old.join(drop, on=key, how="anti")
    new_only = new.join(old, on=key, how="anti").join(q, on=key, how="left")
    add = (new_only.filter(pl.col("q") >= args.add_above)
                   .join(kept.select("match_id"), on="match_id", how="anti")  # record still free
                   .sort("q", descending=True).unique("match_id", keep="first").select(key))
    result = pl.concat([kept, add])
    assert result["match_id"].n_unique() == result.height, "a record would belong to two S1 entities"
    log(f"countries without training data {unseen}: older version {old.height:,} pairs, newer {new.height:,}; "
        f"dropped {drop.height:,} (newer dropped, ours q < {args.drop_below}), added {add.height:,} "
        f"(newer added, ours q >= {args.add_above}) -> {result.height:,}")
    out = Path(args.out) if Path(args.out).is_absolute() else ROOT / args.out
    write_id_lists(result, s1u["s1_id"], out, "matched_entity_ids")


if __name__ == "__main__":
    main()
