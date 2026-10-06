"""Baseline 0: every test S1 entity predicted as a singleton (empty match list).

Stdlib only. Scores exactly the test singleton rate on the leaderboard.
Run from student_resource/:  python src/baseline0.py
"""
import machine_limits  # noqa: F401  -- must stay the first import (caps CPU threads)
import os

TEST_S1 = "dataset/test/test_source1.tsv"
OUT_DIR = "output"


def main():
    with open(TEST_S1, encoding="utf-8") as f:
        next(f)
        ids = [line.split("\t", 1)[0] for line in f if line.strip()]
    assert len(ids) == len(set(ids)), "duplicate S1 ids in test_source1"
    os.makedirs(OUT_DIR, exist_ok=True)
    for name, col in [("matching_results.tsv", "matched_entity_ids"),
                      ("candidate_pairs.tsv", "candidate_entity_ids")]:
        with open(os.path.join(OUT_DIR, name), "w", encoding="utf-8") as f:
            f.write(f"source1_entity_id\t{col}\n")
            f.writelines(f"{i}\t\n" for i in ids)
    print(f"wrote {len(ids)} rows to {OUT_DIR}/")


if __name__ == "__main__":
    main()
