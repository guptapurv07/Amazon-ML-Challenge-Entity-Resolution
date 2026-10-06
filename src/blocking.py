"""Candidate generation (blocking), run independently per country.

Every record gets several cheap keys built from its normalized name/address.
S1 and S2/S3 records sharing a key become candidate pairs; keys whose block is
too large (chains, generic words, malls) are dropped. The union over key
families is the raw candidate set.

"Rare" tokens are ranked by their document frequency among the S1 records of
the country, and tokens that never occur in S1 are skipped: a typo token such
as "wirless" is rare but can never produce a match, so it must not use up a
key slot.

Key families:
  nt   : rare name token x address number        (typos elsewhere, word reorder)
  na   : rare name token x rare address token    (house-number typos / missing)
  an   : address number x rare address token     (name heavily corrupted)
  tt   : the two rarest name tokens              (empty / different address)
  tt   : every pair of the three rarest name tokens (empty / different address;
         a typo in one of the three still leaves a pair)
  cat  : core name with spaces and glued legal words removed
                                                 ("ridgeinitiative.com", "bababusinessprivate")
  cn   : whole core name x house number          (common name, short address:
                                                  "lotus builders" + 5)
  ca   : whole core name x any address word      ("sky marketing" + "chennai")
  acr  : S1 initials x number vs short record name x number
                                                 ("Universal Foods" = "UF", "kpprivate")
  cats : core words in sorted order, joined      (reordered names: "finance my" = "my finance")
  sn   : sorted core x house number              (reordered common names)
  ttn  : pair of rare name tokens x house number (common words, typo in the rest:
                                                  "creative hgitch consultants" + 30)
  nn   : two house numbers of the address        (a different trade name at the
                                                  same detailed address: "624" + "21")
  t1   : one very rare name token alone          (typo in the other word, no address:
                                                  "verran scicntirfic"; blocks of <= 5 S1)
  aa   : pair of the three rarest address tokens (trade name at the same address
                                                  without house numbers)
All frequencies are computed on the data being blocked, so nothing depends on
a particular country.

Memory (8 GB machine): tokens are hashed to u64, token ranking runs 1M records
at a time, and pairs are written to disk per S1 chunk. A pair is stored as one
u64 `pair` = s1_rid * 2^32 + m_rid plus a two-byte family bitmask `bits`
(bit i = FAMILIES[i]). Output: <out_dir>/chunk_<c>.parquet holds the pairs of
the S1 records with s1_rid % N_CHUNKS == c, sorted by `pair`.
"""
import machine_limits  # noqa: F401  -- must stay the first import (caps CPU threads)
import shutil

import polars as pl

from data_io import WORK, country_slug, log

# family -> (max S1 per block, max S1 x S2/S3 pairs per block)
FAMILY_CAPS = {
    "nt": (20, 600),
    "na": (20, 600),
    "an": (10, 200),
    "tt": (20, 400),
    "cat": (20, 400),
    "cn": (20, 600),
    "ca": (20, 600),
    "acr": (10, 200),
    "cats": (20, 400),
    "sn": (20, 600),
    "ttn": (20, 600),
    "nn": (10, 200),
    "t1": (5, 100),
    "aa": (10, 200),
}
# legal words glued to a name ("bababusinessprivate", "privatepromotersmbq")
GLUED_LEGAL = r"^(?:private|pvt|limited|ltd|llp|inc|corp)+|(?:private|pvt|limited|ltd|llp|inc|corp)+$"
FAMILIES = tuple(FAMILY_CAPS)
N_RARE_NAME = 3
N_RARE_ADDR = 2
N_NUMS = 2
N_NUMS_WHOLE = 4  # numbers combined with the whole name (junk prefixes shift the real one)
N_CHUNKS = 8
CHUNK_ROWS = 1_000_000
PAIR_BASE = 1 << 32


def family_bit(fam):
    return 1 << FAMILIES.index(fam)


def encode_pair(s1_rid, m_rid):
    return s1_rid.cast(pl.UInt64) * PAIR_BASE + m_rid.cast(pl.UInt64)


def decode_pairs(df):
    """(pair, ...) -> (s1_rid, m_rid, ...)."""
    return df.with_columns(
        (pl.col("pair") // PAIR_BASE).cast(pl.UInt32).alias("s1_rid"),
        (pl.col("pair") % PAIR_BASE).cast(pl.UInt32).alias("m_rid"),
    ).drop("pair")


def blk_dir(split, country):
    """Where the raw blocking pairs of one (split, country) are written."""
    return WORK / "blk" / split / country_slug(country)


def chunk_path(out_dir, c):
    return out_dir / f"chunk_{c}.parquet"


def _hashed_tokens(df, col, min_len, seed, drop_digit_tokens):
    """(rid, tok) with tok = u64 hash of each distinct token of `col`."""
    keep = pl.col("tok").str.len_chars() >= min_len
    if drop_digit_tokens:
        keep = keep & ~pl.col("tok").str.contains(r"\d")
    parts = []
    for start in range(0, df.height, CHUNK_ROWS):
        parts.append(
            df.slice(start, CHUNK_ROWS)
            .select("rid", pl.col(col).str.split(" ").alias("tok")).explode("tok")
            .filter(keep).select("rid", pl.col("tok").hash(seed=seed)).unique())
    return pl.concat(parts)


def _ranked_tokens(df, col, min_len, seed, drop_digit_tokens, k):
    """Top-k tokens per record, rarest-first by S1 document frequency.
    Tokens absent from S1 are dropped. Runs CHUNK_ROWS records at a time."""
    s1 = df.filter(pl.col("src") == "S1")
    s1_df = _hashed_tokens(s1, col, min_len, seed, drop_digit_tokens).group_by("tok").len("df")
    parts = []
    for start in range(0, df.height, CHUNK_ROWS):
        t = _hashed_tokens(df.slice(start, CHUNK_ROWS), col, min_len, seed, drop_digit_tokens)
        parts.append(t.join(s1_df, on="tok").sort("rid", "df", "tok")
                     .group_by("rid", maintain_order=True).head(k).select("rid", "tok"))
    return pl.concat(parts)


def _numbers(df, k):
    """(rid, num) with the first k distinct numbers of the address (hashed)."""
    return (df.select("rid", pl.col("addr_nums")
                      .list.eval(pl.element().str.strip_chars_start("0"))
                      .list.eval(pl.element().filter(pl.element().str.len_chars() > 0))
                      .list.unique(maintain_order=True).list.head(k).alias("num"))
              .explode("num").drop_nulls()
              .with_columns(pl.col("num").hash(seed=3)))


def prepare_tokens(df):
    """Per-record token material used by the keys (all hashed to u64)."""
    cat = pl.col("name_core").str.replace_all(" ", "").str.replace_all(GLUED_LEGAL, "")
    words = pl.col("name_core").str.split(" ")
    cat_sorted = words.list.sort().list.join("").str.replace_all(GLUED_LEGAL, "")
    # S1 side: initials of multi-word names; other side: short names that could be acronyms
    acr = (pl.when(pl.col("src") == "S1")
             .then(pl.when(words.list.len() >= 2)
                   .then(words.list.eval(pl.element().str.slice(0, 1)).list.join("")))
             .otherwise(pl.when(cat.str.len_chars().is_between(2, 5)).then(cat)))
    return {
        "name": _ranked_tokens(df, "name_core", 2, 1, False, N_RARE_NAME),
        "addr": _ranked_tokens(df, "addr_norm", 3, 2, True, N_RARE_ADDR).rename({"tok": "atok"}),
        "num": _numbers(df, N_NUMS),
        "num4": _numbers(df, N_NUMS_WHOLE),
        "cat": df.select("rid", cat.alias("c")).filter(pl.col("c") != "").select(
            "rid", pl.col("c").hash(seed=16)),
        "cats": df.filter(words.list.len() >= 2).select("rid", cat_sorted.alias("c"))
                  .filter(pl.col("c") != "").select("rid", pl.col("c").hash(seed=21)),
        "acr": df.select("rid", acr.alias("a")).drop_nulls().select(
            "rid", pl.col("a").hash(seed=17).alias("c")),
        "words": _hashed_tokens(df, "addr_norm", 3, 2, True).rename({"tok": "atok"}),
        "addr3": _ranked_tokens(df, "addr_norm", 3, 2, True, 3),
    }


def _token_pairs(name):
    """(rid, tok, tok_b): every unordered pair of a record's ranked rare name tokens."""
    return name.join(name, on="rid", suffix="_b").filter(pl.col("tok") < pl.col("tok_b"))


def family_keys(df, toks, fam):
    """(rid, h) hashed keys of one family."""
    name, addr, num = toks["name"], toks["addr"], toks["num"]
    if fam == "nt":
        k = name.join(num, on="rid").select("rid", pl.struct("tok", "num").hash(seed=11).alias("h"))
    elif fam == "na":
        k = name.join(addr, on="rid").select("rid", pl.struct("tok", "atok").hash(seed=12).alias("h"))
    elif fam == "an":
        k = num.join(addr, on="rid").select("rid", pl.struct("num", "atok").hash(seed=13).alias("h"))
    elif fam == "tt":
        k = _token_pairs(name).select("rid", pl.struct("tok", "tok_b").hash(seed=14).alias("h"))
    elif fam == "ttn":
        k = _token_pairs(name).join(num, on="rid").select(
            "rid", pl.struct("tok", "tok_b", "num").hash(seed=23).alias("h"))
    elif fam == "cats":
        k = toks["cats"].select("rid", pl.col("c").alias("h"))
    elif fam == "sn":
        k = toks["cats"].join(toks["num4"], on="rid").select(
            "rid", pl.struct("c", "num").hash(seed=22).alias("h"))
    elif fam == "t1":
        k = name.group_by("rid", maintain_order=True).head(2).select(
            "rid", pl.col("tok").hash(seed=25).alias("h"))
    elif fam == "aa":
        k = _token_pairs(toks["addr3"]).select("rid", pl.struct("tok", "tok_b").hash(seed=26).alias("h"))
    elif fam == "nn":
        n4 = toks["num4"]
        k = (n4.join(n4, on="rid", suffix="_b").filter(pl.col("num") < pl.col("num_b"))
               .select("rid", pl.struct("num", "num_b").hash(seed=24).alias("h")))
    elif fam == "cat":
        k = toks["cat"].select("rid", pl.col("c").alias("h"))
    elif fam == "cn":
        k = toks["cat"].join(toks["num4"], on="rid").select(
            "rid", pl.struct("c", "num").hash(seed=18).alias("h"))
    elif fam == "ca":
        k = toks["cat"].join(toks["words"], on="rid").select(
            "rid", pl.struct("c", "atok").hash(seed=19).alias("h"))
    elif fam == "acr":
        k = toks["acr"].join(toks["num4"], on="rid").select(
            "rid", pl.struct("c", "num").hash(seed=20).alias("h"))
    else:
        raise ValueError(fam)
    return k.unique()


def generate_candidates(df, out_dir, families=FAMILIES, n_chunks=N_CHUNKS):
    """Write the raw candidate pairs of ONE country to out_dir/chunk_<c>.parquet.

    df: normalized records of one country with a unique integer `rid` column.
    Returns the total number of pairs.
    """
    shutil.rmtree(out_dir, ignore_errors=True)
    tmp = out_dir / "tmp"
    tmp.mkdir(parents=True)
    toks = prepare_tokens(df)
    is_s1 = df.select("rid", (pl.col("src") == "S1").alias("is_s1"))
    for fam in families:
        max_s1, max_pairs = FAMILY_CAPS[fam]
        keys = family_keys(df, toks, fam).join(is_s1, on="rid")
        ok = keys.group_by("h").agg(
            pl.col("is_s1").sum().alias("n1"), (~pl.col("is_s1")).sum().alias("n2")
        ).filter(
            (pl.col("n1") > 0) & (pl.col("n2") > 0) & (pl.col("n1") <= max_s1)
            & (pl.col("n1") * pl.col("n2") <= max_pairs)
        ).select("h")
        keys = keys.join(ok, on="h")
        left = keys.filter("is_s1").select(pl.col("rid").alias("s1_rid"), "h")
        right = keys.filter(~pl.col("is_s1")).select(pl.col("rid").alias("m_rid"), "h")
        del keys, ok
        n_fam = 0
        for c in range(n_chunks):
            p = (left.filter(pl.col("s1_rid") % n_chunks == c).join(right, on="h")
                 .select(encode_pair(pl.col("s1_rid"), pl.col("m_rid")).alias("pair")).unique())
            p.with_columns(pl.lit(family_bit(fam), dtype=pl.UInt16).alias("bits")).write_parquet(
                tmp / f"{fam}_{c}.parquet")
            n_fam += p.height
            del p
        log(f"  family {fam}: {n_fam} pairs")
        del left, right
    del toks, is_s1
    total = 0
    for c in range(n_chunks):  # families only overlap within the same S1 chunk
        merged = (pl.concat([pl.read_parquet(tmp / f"{fam}_{c}.parquet") for fam in families])
                  .group_by("pair").agg(pl.col("bits").sum().cast(pl.UInt16)).sort("pair"))
        merged.write_parquet(chunk_path(out_dir, c))
        total += merged.height
        del merged
    shutil.rmtree(tmp)
    return total
