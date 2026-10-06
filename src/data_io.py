"""Reading and writing the challenge TSV files.

Named data_io.py (not io.py) so it never shadows the stdlib `io` module when
scripts are run as `python src/<script>.py`.

All source files are tab-separated with standard CSV quoting on a few fields,
so polars' default quote handling is used and row counts are asserted.
"""
import machine_limits  # noqa: F401  -- must stay the first import (caps CPU threads)
import hashlib
import json
import re
import time
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "dataset"
WORK = ROOT / "work"
OUTPUT = ROOT / "output"
NORM = WORK / "norm"          # normalized records: norm/<split>/<country>/s<source>.parquet
GT_PAIRS = WORK / "gt_pairs.parquet"

SOURCE_COLS = ["entity_id", "business_name", "business_address", "country"]


def country_slug(country):
    """Folder name for a country label. Labels are open-ended, so the name is
    sanitized and suffixed with a short hash to keep distinct labels distinct."""
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", country).strip("_") or "country"
    return f"{safe}-{hashlib.md5(country.encode()).hexdigest()[:6]}"


def _country_index_path(split):
    return NORM / split / "countries.json"


def write_norm_part(df, split, source):
    """Write normalized records of one source, one parquet file per country."""
    index_path = _country_index_path(split)
    index = json.loads(index_path.read_text()) if index_path.exists() else {}
    for country in df["country"].unique().sort().to_list():
        out = NORM / split / country_slug(country) / f"s{source}.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        df.filter(pl.col("country") == country).write_parquet(out)
        index[country] = country_slug(country)
    index_path.write_text(json.dumps(index, indent=1, sort_keys=True))


def list_countries(split):
    """Countries present in a split (read from the data, never hard-coded)."""
    return sorted(json.loads(_country_index_path(split).read_text()))


def load_norm(split, country, columns=None):
    """Normalized records (all sources) of one country."""
    folder = NORM / split / country_slug(country)
    return pl.concat([pl.read_parquet(f, columns=columns)
                      for f in sorted(folder.glob("s*.parquet"))])


def load_gt_pairs():
    """Cached (s1_id, match_id) train label pairs written by prepare.py."""
    return pl.read_parquet(GT_PAIRS)


def count_data_lines(path):
    """Number of lines in a file minus the header."""
    n = 0
    with open(path, "rb") as f:
        while chunk := f.read(1 << 24):
            n += chunk.count(b"\n")
    return n - 1


def _read_tsv(path, expected_cols):
    df = pl.read_csv(
        path,
        separator="\t",
        infer_schema=False,  # every column as string
        missing_utf8_is_empty_string=True,
        quote_char='"',
    )
    assert df.columns == expected_cols, f"{path}: unexpected columns {df.columns}"
    n_lines = count_data_lines(path)
    assert df.height == n_lines, f"{path}: parsed {df.height} rows but file has {n_lines}"
    return df


def source_path(split, source):
    return DATA / split / f"{split}_source{source}.tsv"


def read_source(split, source):
    """Load one source file (split in {'train','test'}, source in {1,2,3})."""
    df = _read_tsv(source_path(split, source), SOURCE_COLS)
    return df.with_columns(pl.col(c).str.strip_chars() for c in SOURCE_COLS)


def read_sources(split, sources=(1, 2, 3)):
    """Concatenate several sources; entity_id prefix identifies the source."""
    return pl.concat([read_source(split, s) for s in sources])


def read_ground_truth():
    """Train labels as a pair table (s1_id, match_id). Singletons have no rows."""
    gt = _read_tsv(DATA / "train" / "train_ground_truth.tsv",
                   ["source1_entity_id", "matched_entity_ids"])
    return (
        gt.rename({"source1_entity_id": "s1_id"})
        .with_columns(pl.col("matched_entity_ids").str.split(","))
        .explode("matched_entity_ids")
        .rename({"matched_entity_ids": "match_id"})
        .filter(pl.col("match_id").str.len_chars() > 0)
    )


def write_id_lists(pairs, s1_ids, path, list_col):
    """Write one row per S1 id with a comma-joined, sorted, de-duplicated id list.

    pairs: DataFrame with columns (s1_id, match_id).
    s1_ids: every S1 id that must appear (empty list for ones without pairs).
    """
    grouped = (
        pairs.select("s1_id", "match_id")
        .unique()
        .sort("s1_id", "match_id")
        .group_by("s1_id", maintain_order=True)
        .agg(pl.col("match_id").str.join(",").alias(list_col))
    )
    out = (
        pl.DataFrame({"s1_id": s1_ids}, schema={"s1_id": pl.String})
        .join(grouped, on="s1_id", how="left")
        .with_columns(pl.col(list_col).fill_null(""))
        .rename({"s1_id": "source1_entity_id"})
    )
    assert out.height == len(s1_ids)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    out.write_csv(path, separator="\t", quote_style="never")
    return out


def log(msg):
    """Timestamped, flushed progress line (readable while a job runs in background)."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
