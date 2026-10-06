"""Team ensemble: this pipeline's final probabilities + a teammate pipeline's match list.

Both pipelines run on the same test data (only the provided files). This step combines
them; it never looks at labels and adds no pair by hand.

Inputs
  --ours DIR          stage2.py output written with --save-test-probs: test_probs.parquet
                      (every candidate pair with its final probability q and threshold)
  --mate FILE         the teammate pipeline's matching_results.tsv
  --mate-candidates FILE  (optional) the teammate pipeline's candidate_pairs.tsv
  --mate-unseen       countries without training data: the teammate's matches as they are
Rule
  * a candidate pair gets posterior odds = our odds x VOTE_LR if the teammate matched it,
    our odds / VOTE_LR if not (his decision as evidence of fixed strength): where our model
    is sure it decides, near our threshold his vote decides;
  * a pair he matched that is not among our candidates (his blocking found it, ours did not)
    gets probability OUTSIDE_P;
  * the usual one-to-one decision (each S2/S3 record under its most probable S1) at our
    threshold; with --mate-unseen, countries without training data (test countries that
    are not train countries) keep his matches instead.
Outputs (--out-dir, default output/)
  matching_results.tsv  the merged matches
  candidate_pairs.tsv   the candidate set both pipelines scored: our candidates, his
                        candidates (if given) and every pair he matched, so every match
                        is a candidate, as the rules require

Run from student_resource/:
  python src/team_merge.py --ours work/sub_v6_final_ce3fr_probs --mate <teammate matching TSV>
                           [--mate-candidates <teammate candidate TSV>] --mate-unseen
"""
import machine_limits  # noqa: F401  -- must stay the first import (caps CPU threads)
import argparse
from pathlib import Path

import numpy as np
import polars as pl

from data_io import OUTPUT, ROOT, list_countries, log, write_id_lists
from decide import decide
from predict import test_s1_ids

VOTE_LR = 3.0
OUTSIDE_P = 0.9


def id_pairs(path, list_col):
    """(s1_id, match_id) pairs of a submission-style TSV (source1_entity_id, <list_col>)."""
    m = pl.read_csv(path, separator="\t", schema={"s1_id": pl.String, list_col: pl.String})
    return (m.with_columns(pl.col(list_col).fill_null("").str.split(",")).explode(list_col)
             .filter(pl.col(list_col) != "").rename({list_col: "match_id"}).unique())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", required=True, help="stage2.py --save-test-probs output folder")
    ap.add_argument("--mate", required=True, help="teammate's matching_results.tsv")
    ap.add_argument("--mate-candidates", help="teammate's candidate_pairs.tsv (optional)")
    ap.add_argument("--mate-unseen", action="store_true",
                    help="countries without training data: the teammate's matches as they are")
    ap.add_argument("--mate-unseen-file",
                    help="with --mate-unseen: take the matches for countries without training data from this "
                         "file (e.g. an earlier version of the teammate's pipeline) instead of --mate")
    ap.add_argument("--vote-lr", type=float, default=VOTE_LR, help="strength of his vote (odds factor)")
    ap.add_argument("--unseen-vote-lr", type=float, default=None,
                    help="strength of his vote in countries without training data (default: --vote-lr)")
    ap.add_argument("--out-dir", default=str(OUTPUT))
    args = ap.parse_args()
    if args.mate_unseen_file and not args.mate_unseen:
        ap.error("--mate-unseen-file needs --mate-unseen")
    out_dir = Path(args.out_dir) if Path(args.out_dir).is_absolute() else ROOT / args.out_dir

    pr = pl.read_parquet(Path(args.ours) / "test_probs.parquet")  # s1_id, match_id, q, thr
    thr = float(pr["thr"].max())
    mate = id_pairs(args.mate, "matched_entity_ids").with_columns(pl.lit(True).alias("vote"))
    s1 = test_s1_ids()
    unseen = [c for c in list_countries("test") if c not in list_countries("train")]
    country = s1.select(pl.col("entity_id").alias("s1_id"), "country")
    d = (pr.join(mate, on=["s1_id", "match_id"], how="left").with_columns(pl.col("vote").fill_null(False))
           .join(country, on="s1_id", how="left"))
    q = d["q"].clip(1e-6, 1 - 1e-6).to_numpy()
    lr = np.where(d["country"].is_in(unseen).to_numpy(),
                  args.vote_lr if args.unseen_vote_lr is None else args.unseen_vote_lr, args.vote_lr)
    odds = q / (1 - q) * np.where(d["vote"].to_numpy(), lr, 1 / lr)
    d = d.select("s1_id", "match_id").with_columns(pl.Series("p", odds / (1 + odds)))
    outside = mate.join(pr, on=["s1_id", "match_id"], how="anti").select(
        "s1_id", "match_id", pl.lit(OUTSIDE_P).alias("p"))
    log(f"{d.height} candidate pairs of ours; teammate matched {mate.height} pairs, "
        f"{outside.height} of them outside our candidates; threshold {thr}")
    matches = decide(pl.concat([d, outside]), thr).select("s1_id", "match_id")

    mate_cands = mate.select("s1_id", "match_id")
    if args.mate_unseen:
        in_unseen = pl.col("country").is_in(unseen)
        unseen_mate = (id_pairs(args.mate_unseen_file, "matched_entity_ids") if args.mate_unseen_file
                       else mate.select("s1_id", "match_id"))
        mine = matches.join(country, on="s1_id").filter(~in_unseen).select("s1_id", "match_id")
        his = unseen_mate.join(country, on="s1_id").filter(in_unseen).select("s1_id", "match_id")
        matches = pl.concat([mine, his])
        log(f"countries without training data {unseen}: the teammate's {his.height} matches"
            + (f" from {Path(args.mate_unseen_file).name}" if args.mate_unseen_file else ""))
        if args.mate_unseen_file:  # there, only the matches actually used are candidates
            mate_cands = pl.concat([mate_cands.join(country, on="s1_id").filter(~in_unseen)
                                    .select("s1_id", "match_id"), his])
    cands = [pr.select("s1_id", "match_id"), mate_cands]
    if args.mate_candidates:
        cands.append(id_pairs(args.mate_candidates, "candidate_entity_ids"))
    cands = pl.concat(cands).unique()
    missing = matches.join(cands, on=["s1_id", "match_id"], how="anti").height
    assert missing == 0, f"{missing} matches are not candidates"
    write_id_lists(cands, s1["entity_id"], out_dir / "candidate_pairs.tsv", "candidate_entity_ids")
    write_id_lists(matches, s1["entity_id"], out_dir / "matching_results.tsv", "matched_entity_ids")
    log(f"wrote {out_dir}: {matches.height} matches, {cands.height} candidates "
        f"({cands.height / s1.height:.2f} per S1)")


if __name__ == "__main__":
    main()
