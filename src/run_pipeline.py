"""End-to-end pipeline: raw TSVs -> output/matching_results.tsv + output/candidate_pairs.tsv.

Run from student_resource/:
  python src/run_pipeline.py [--mate <teammate's matching_results.tsv>]

The CPU part runs here (tested on an 8 GB Apple M2, 4 threads). The three cross-encoder
rounds need a GPU and run as notebooks (Google Colab / Kaggle). The command is
re-entrant: every finished step leaves a marker in work/.done/; when cross-encoder scores
are missing it stops and prints which notebook to run and which files to copy into
work/ce/; running the same command again continues where it stopped.

Phase 1, CPU (about 3 h):
  1. prepare.py                        normalize every record; token maps learned from train pairs
  2. candidates.py --split train       blocking + cheap scores of every raw pair (cached)
  3. cut_model.py                      learned one-to-one cut, cross-fitted on train
  4. candidates.py                     final candidate sets (learned cut), train and test
  5. train.py                          pair model: features, 4-fold CV, leave-one-country-out
  6. predict.py                        test probabilities (work/test_predictions.parquet)
  7. ce_export.py                      cross-encoder files for rounds 1-3 -> work/ce/
Phase 2, GPU notebooks (upload work/ce/ files to the Google Drive folder MyDrive/er_ce):
  colab/cross_encoder.ipynb            round 1: fine-tune xlm-roberta-base (MIT, 278M)
  colab/cross_encoder_round2.ipynb     round 2: more train pairs + confident pairs of countries
                                       without training data; scores every hard pair
  colab/cross_encoder_round3.ipynb     round 3: countries without training data; scores their
                                       hard test pairs
  -> copy ce2_scores_*.parquet and ce3_scores_test.parquet back into work/ce/
Phase 3, CPU (minutes):
  8. stage2.py                         pair model + cross-encoders -> final probabilities
  9. team_merge.py (with --mate)       combine with the teammate pipeline's matches
 10. utils/validate_submission.py      the organizers' validator, if present: must print PASS
"""
import machine_limits  # noqa: F401  -- must stay the first import (caps CPU threads)
import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

from data_io import ROOT, WORK, log

CE = WORK / "ce"
DONE = WORK / ".done"
STAGE2_DIR = "work/sub_final"
GPU_FILES = ["ce2_scores_oof.parquet", "ce2_scores_test.parquet", "ce2_scores_oof_loco.parquet",
             "ce3_scores_test.parquet"]

PHASE1 = [
    ("prepare", ["src/prepare.py"]),
    ("candidates_train_raw", ["src/candidates.py", "--split", "train"]),
    ("cut_model", ["src/cut_model.py"]),
    ("candidates_train", ["src/candidates.py", "--split", "train", "--reuse-blocking", "--reuse-scores"]),
    ("candidates_test", ["src/candidates.py", "--split", "test"]),
    ("train", ["src/train.py"]),
    ("predict", ["src/predict.py", "--out-dir", "work/sub_stage1"]),
    ("ce_round1", ["src/ce_export.py"]),
    ("ce_round2_train", ["src/ce_export.py", "--round2", "train"]),
    ("ce_round2_score", ["src/ce_export.py", "--round2", "score"]),
    ("ce_round3_train", ["src/ce_export.py", "--round3", "train"]),
    ("ce_round3_score", ["src/ce_export.py", "--round3", "score"]),
]


def run_step(name, args):
    if (DONE / name).exists():
        log(f"--- {name}: done earlier, skipped")
        return
    log(f"=== {name}: {' '.join(args)}")
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True)
    DONE.mkdir(parents=True, exist_ok=True)
    (DONE / name).touch()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mate", help="the teammate pipeline's matching_results.tsv (team merge)")
    ap.add_argument("--mate-unseen-file",
                    help="countries without training data: the teammate's matches from this file instead (team_merge.py)")
    ap.add_argument("--vote-lr", default="8", help="strength of the teammate's vote (team_merge.py)")
    ap.add_argument("--out-dir", default="output", help="where the two final TSV files go")
    ap.add_argument("--fresh", action="store_true", help="forget the step markers, start from scratch")
    args = ap.parse_args()
    if args.fresh and DONE.exists():
        shutil.rmtree(DONE)
    t0 = time.time()

    for name, step in PHASE1:
        run_step(name, step)

    missing = [f for f in GPU_FILES if not (CE / f).exists()]
    if missing:
        log("Phase 2 (GPU) needed. Upload work/ce/ce_train.parquet, ce2_train.parquet, "
            "ce2_oof.parquet, ce2_test.parquet, ce2_oof_loco.parquet, ce3_train.parquet and "
            "ce3_test.parquet to MyDrive/er_ce, run colab/cross_encoder.ipynb, then "
            "colab/cross_encoder_round2.ipynb, then colab/cross_encoder_round3.ipynb, copy the "
            f"score files back into work/ce/ (missing: {', '.join(missing)}) and run this command again.")
        return

    run_step("stage2", ["src/stage2.py", "--stage2-unseen", "--ce-prefix", "ce2", "--ce-unseen-prefix", "ce3",
                        "--save-test-probs", "--out-dir", STAGE2_DIR])
    out_dir = Path(args.out_dir) if Path(args.out_dir).is_absolute() else ROOT / args.out_dir
    if args.mate:
        log("=== team_merge")
        unseen_file = ["--mate-unseen-file", args.mate_unseen_file] if args.mate_unseen_file else []
        subprocess.run([sys.executable, "src/team_merge.py", "--ours", STAGE2_DIR, "--mate", args.mate,
                        "--mate-unseen", *unseen_file, "--vote-lr", args.vote_lr, "--out-dir", str(out_dir)],
                       cwd=ROOT, check=True)
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
        for f in ("matching_results.tsv", "candidate_pairs.tsv"):
            shutil.copy2(ROOT / STAGE2_DIR / f, out_dir / f)
    validator = ROOT / "utils" / "validate_submission.py"  # the organizers' script, not part of this repository
    if validator.exists():
        log("=== validator")
        subprocess.run([sys.executable, str(validator),
                        "--matching", str(out_dir / "matching_results.tsv"),
                        "--candidate", str(out_dir / "candidate_pairs.tsv"), "--test-dir", "dataset/test"],
                       cwd=ROOT, check=True)
    else:
        log("utils/validate_submission.py (the organizers' validator) not found: validation skipped")
    log(f"pipeline finished in {(time.time() - t0) / 60:.1f} min -> {out_dir}")


if __name__ == "__main__":
    main()
