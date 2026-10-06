"""Normalize all source records and cache them as parquet in work/.

One source file at a time, to keep memory low on an 8 GB machine.

Outputs:
  work/norm/<split>/<country>/s<source>.parquet   normalized records
  work/norm/<split>/countries.json                country label -> folder
  work/token_maps.json                            learned token maps (see normalize.mine_token_maps)
  work/gt_pairs.parquet                           train labels as (s1_id, match_id)
"""
import machine_limits  # noqa: F401  -- must stay the first import (caps CPU threads)
import json
import shutil

import polars as pl

from data_io import GT_PAIRS, NORM, WORK, log, read_ground_truth, read_source, write_norm_part
from normalize import apply_learned_maps, mine_name_dict, mine_token_maps, normalize_records

SPLITS = ("train", "test")
SOURCES = (1, 2, 3)


def main():
    stage = WORK / "stage"
    shutil.rmtree(NORM, ignore_errors=True)
    stage.mkdir(parents=True, exist_ok=True)

    # 1) base normalization, one source file at a time
    for split in SPLITS:
        for source in SOURCES:
            df = normalize_records(read_source(split, source))
            df.write_parquet(stage / f"{split}_s{source}.parquet")
            log(f"normalized {split} source {source}: {df.height} rows")
            del df

    # 2) token maps learned from matched train pairs (labels only, no outside data)
    gt = read_ground_truth()
    gt.write_parquet(GT_PAIRS)
    train = pl.concat([
        pl.read_parquet(stage / f"train_s{s}.parquet",
                        columns=["entity_id", "src", "had_nonascii", "name_norm", "addr_norm"])
        for s in SOURCES])
    maps = {"script": mine_name_dict(train, gt)}
    maps["name"], maps["addr"] = mine_token_maps(train, gt)
    del train, gt
    with open(WORK / "token_maps.json", "w") as f:
        json.dump(maps, f, indent=0, sort_keys=True, ensure_ascii=False)
    log("learned token maps: " + ", ".join(f"{k} {len(v)}" for k, v in maps.items()))

    # 3) apply the maps, derive core names, split by country
    for split in SPLITS:
        for source in SOURCES:
            path = stage / f"{split}_s{source}.parquet"
            write_norm_part(apply_learned_maps(pl.read_parquet(path), maps), split, source)
            path.unlink()
    stage.rmdir()
    for split in SPLITS:
        log(f"{split} countries: {json.loads((NORM / split / 'countries.json').read_text())}")


if __name__ == "__main__":
    main()
