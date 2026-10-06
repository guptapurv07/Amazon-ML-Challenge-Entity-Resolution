"""Pair features for candidate pairs, one country at a time, in chunks.

Records are addressed by `rid` = row position in the country's record table,
so both sides of a pair are fetched with `gather` (fast, order-preserving).

Feature groups:
  name    : rapidfuzz ratio / token-set / token-sort / partial / Jaro-Winkler
            on normalized and core names, plus the space-free "cat" form
  address : token-set and plain ratio on normalized addresses, house-number
            overlap / conflict, empty-address flags
  blocking: which key families produced the pair
  legal   : legal-form same / partial / conflict (Private Limited vs LLP is a
            different company) / missing on one side
  context : how many S1 share the core name (chains) or the address (malls); how
            many S2/S3 records carry each side's exact core name (decoy vs typo)
Cheap scores (name_tsr, cat_ratio, addr_tsr) and the competition features
(count / rank / gap / margin among the raw candidates of the S1 and of the
S2/S3 record) come with the candidate table built by candidates.py.
No country feature: every statistic is computed within the data being scored.
"""
import machine_limits  # noqa: F401  -- must stay the first import (caps CPU threads)
import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from rapidfuzz.process import cpdist

from blocking import FAMILIES, family_bit
from machine_limits import N_THREADS

CHUNK_PAIRS = 1_000_000
RECORD_COLS = ["entity_id", "src", "had_nonascii", "name_norm", "name_core",
               "addr_norm", "addr_nums"]


# Legal-form tokens after normalize.NAME_CANON. "pvt" also counts as "ltd": noise
# drops "Limited" ("Rely Club Private") or misspells "Private" ("Pritaea Limited").
LEGAL_FORMS = ["pvt", "ltd", "inc", "corp", "co", "cie", "llc", "llp", "lp", "plc", "pllc",
               "pc", "sarl", "sas", "sasu", "eurl", "sa", "sci", "snc", "scop"]


def legal_form(name_norm):
    """Sorted set of legal-form tokens of a normalized name, e.g. ["ltd", "pvt"], ["llp"]."""
    form = (name_norm.str.split(" ")
            .list.eval(pl.element().filter(pl.element().is_in(LEGAL_FORMS))).list.join(" "))
    form = pl.when(form.str.contains(r"\bpvt\b")).then(form + " ltd").otherwise(form)
    return (form.str.split(" ").list.eval(pl.element().filter(pl.element().str.len_chars() > 0))
            .list.unique().list.sort())


def record_table(df):
    """Per-record columns used by the pair features. `df` must have `rid` = row index."""
    s1 = df.filter(pl.col("src") == "S1")
    core_freq = s1.group_by("name_core").len("s1_same_core")
    addr_freq = s1.filter(pl.col("addr_norm") != "").group_by("addr_norm").len("s1_same_addr")
    # how many S2/S3 records carry exactly this core name: a decoy entity's name tends
    # to repeat across its own records, a one-off typo of an S1 name does not
    support = df.filter(pl.col("src") != "S1").group_by("name_core").len("core_support")
    return (
        df.with_columns(
            pl.col("name_core").str.replace_all(" ", "").alias("cat"),
            pl.col("addr_nums").list.eval(pl.element().str.strip_chars_start("0"))
              .list.eval(pl.element().filter(pl.element().str.len_chars() > 0))
              .list.unique(maintain_order=True).alias("nums"),
            (pl.col("src") == "S2").alias("is_s2"),
            (pl.col("addr_norm") == "").alias("addr_empty"),
            pl.col("name_core").str.split(" ").list.len().alias("n_core_tok"),
            legal_form(pl.col("name_norm")).alias("legal"),
            pl.col("name_norm").str.contains(r"\d{5,}").alias("name_digits"),
        )
        .join(core_freq, on="name_core", how="left")
        .join(addr_freq, on="addr_norm", how="left")
        .join(support, on="name_core", how="left")
        .with_columns(pl.col("s1_same_core", "s1_same_addr", "core_support").fill_null(0))
        .sort("rid")
        .with_columns(
            pl.col("nums").list.join(" ").alias("numstr"),
            pl.when(pl.col("n_core_tok") >= 2)
              .then(pl.col("name_core").str.split(" ").list.eval(pl.element().str.slice(0, 1))
                    .list.join(""))
              .otherwise(pl.lit("")).alias("initials"),
        )
        .select("rid", "name_norm", "name_core", "cat", "addr_norm", "nums", "is_s2",
                "had_nonascii", "addr_empty", "n_core_tok", "s1_same_core", "s1_same_addr",
                "legal", "name_digits", "core_support", "numstr", "initials")
    )


def _sim(a, b, scorer):
    return cpdist(a, b, scorer=scorer, workers=N_THREADS, dtype=np.float32)


def _side(rec, rids, suffix):
    return rec[rids.to_numpy()].drop("rid").rename(lambda c: f"{c}{suffix}")


def _num_shift(n1, n2):
    """How far apart the address numbers are, as numbers. A look-alike record can keep
    the S1 name and street a few doors away (house number +1 .. +21), while noise
    changes digits instead (1370 vs 137, 4342 vs 4340).
      num_first_diff  record's first number minus the S1's first number
      num_min_shift   smallest nonzero |difference| over all pairs of their numbers
    NaN when a side has no number (or, for num_min_shift, all numbers agree)."""
    first = np.full(len(n1), np.nan, dtype=np.float32)
    shift = np.full(len(n1), np.nan, dtype=np.float32)
    for i, (a, b) in enumerate(zip(n1, n2)):
        if not a or not b:
            continue
        xa, xb = [int(x[:12]) for x in a], [int(x[:12]) for x in b]
        first[i] = xb[0] - xa[0]
        d = [abs(u - v) for u in xa for v in xb if u != v]
        if d:
            shift[i] = min(d)
    return first, shift


def _u8(x):
    return pl.Series(np.rint(x).astype(np.uint8))


def cheap_scores(pairs, rec):
    """Cheap similarity for EVERY raw blocking pair, used to cut the candidate set:
      name_tsr   token-set ratio of core names (word order / extra words tolerant)
      cat_ratio  ratio of core names with spaces removed ("veteranscharities.com")
      addr_tsr   token-set ratio of normalized addresses
      num_tsr    token-set ratio of the address numbers (position-free)
      acr        100 when the record's name is the S1 name's initials ("UF")
    Stored as UInt8 (0-100): there are tens of millions of raw pairs per country."""
    cols = ["name_core", "cat", "addr_norm", "numstr", "initials"]
    out = []
    for start in range(0, pairs.height, CHUNK_PAIRS):
        p = pairs.slice(start, CHUNK_PAIRS)
        a = rec.select(cols)[p["s1_rid"].to_numpy()]
        b = rec.select(cols)[p["m_rid"].to_numpy()]
        acr = ((a["initials"] != "") & (a["initials"] == b["cat"])).to_numpy()
        out.append(p.with_columns(
            _u8(_sim(a["name_core"].to_list(), b["name_core"].to_list(), fuzz.token_set_ratio))
            .alias("name_tsr"),
            _u8(_sim(a["cat"].to_list(), b["cat"].to_list(), fuzz.ratio)).alias("cat_ratio"),
            _u8(_sim(a["addr_norm"].to_list(), b["addr_norm"].to_list(), fuzz.token_set_ratio))
            .alias("addr_tsr"),
            _u8(_sim(a["numstr"].to_list(), b["numstr"].to_list(), fuzz.token_set_ratio))
            .alias("num_tsr"),
            pl.Series("acr", (acr * 100).astype(np.uint8)),
        ))
    return pl.concat(out)


def pair_features(p, rec):
    """Full feature table for a chunk of pairs (with base/competition columns)."""
    a = _side(rec, p["s1_rid"], "_1")
    b = _side(rec, p["m_rid"], "_2")
    n1, n2 = a["name_norm_1"].to_list(), b["name_norm_2"].to_list()
    c1, c2 = a["name_core_1"].to_list(), b["name_core_2"].to_list()
    d1, d2 = a["addr_norm_1"].to_list(), b["addr_norm_2"].to_list()
    feats = pl.DataFrame({
        "name_ratio": _sim(n1, n2, fuzz.ratio),
        "core_ratio": _sim(c1, c2, fuzz.ratio),
        "core_tsort": _sim(c1, c2, fuzz.token_sort_ratio),
        "core_partial": _sim(c1, c2, fuzz.partial_ratio),
        "core_jw": _sim(c1, c2, JaroWinkler.normalized_similarity),
        "addr_ratio": _sim(d1, d2, fuzz.ratio),
        "addr_partial": _sim(d1, d2, fuzz.partial_ratio),
    })
    nums = pl.DataFrame({"n1": a["nums_1"], "n2": b["nums_2"]}).select(
        pl.col("n1").list.set_intersection("n2").list.len().alias("num_common"),
        pl.col("n1").list.len().alias("num_n1"),
        pl.col("n2").list.len().alias("num_n2"),
        (pl.col("n1").list.first() == pl.col("n2").list.first()).fill_null(False).alias("num_first_eq"),
        # position-free: junk prefixes ("Door No 238 1464", "#595 269") shift the first number
        pl.when((pl.col("n1").list.len() > 0) & (pl.col("n2").list.len() > 0))
          .then(pl.col("n2").list.contains(pl.col("n1").list.first())).alias("s1_num_in_m"),
        pl.when((pl.col("n1").list.len() > 0) & (pl.col("n2").list.len() > 0))
          .then(pl.col("n1").list.contains(pl.col("n2").list.first())).alias("m_num_in_s1"),
        pl.col("n1").list.first().fill_null("").alias("f1"),
        pl.col("n2").list.first().fill_null("").alias("f2"),
    ).with_columns(
        (pl.col("num_common") / (pl.col("num_n1") + pl.col("num_n2") - pl.col("num_common")))
        .fill_nan(None).alias("num_jacc"),
        ((pl.col("num_n1") > 0) & (pl.col("num_n2") > 0) & (pl.col("num_common") == 0))
        .alias("num_conflict"),
    )
    # similarity of the first numbers (1370 vs 137, 4342 vs 4340); NaN when one is missing
    first_ratio = _sim(nums["f1"].to_list(), nums["f2"].to_list(), fuzz.ratio)
    first_ratio[(nums["f1"] == "").to_numpy() | (nums["f2"] == "").to_numpy()] = np.nan
    first_diff, min_shift = _num_shift(a["nums_1"].to_list(), b["nums_2"].to_list())
    nums = nums.drop("f1", "f2").with_columns(pl.Series("num_first_ratio", first_ratio),
                                              pl.Series("num_first_diff", first_diff),
                                              pl.Series("num_min_shift", min_shift))
    # conflict = both names carry a legal form and they share nothing (Pvt Ltd vs LLP);
    # partial = they overlap but differ (Ltd vs Pvt Ltd, often a misspelt "Private")
    legal = pl.DataFrame({"l1": a["legal_1"], "l2": b["legal_2"]}).select(
        pl.col("l1").list.len().alias("n1"), pl.col("l2").list.len().alias("n2"),
        pl.col("l1").list.set_intersection("l2").list.len().alias("ni"),
    ).select(
        ((pl.col("n1") == pl.col("n2")) & (pl.col("ni") == pl.col("n1"))).alias("legal_same"),
        ((pl.col("n1") > 0) & (pl.col("n2") > 0) & (pl.col("ni") == 0)).alias("legal_conflict"),
        ((pl.col("n1") > 0) & (pl.col("n2") > 0) & (pl.col("ni") > 0)
         & ((pl.col("ni") < pl.col("n1")) | (pl.col("ni") < pl.col("n2")))).alias("legal_partial"),
        ((pl.col("n1") == 0) != (pl.col("n2") == 0)).alias("legal_one_missing"),
    )
    bits = pl.DataFrame([
        ((p["bits"] & family_bit(f)) > 0).alias(f"blk_{f}") for f in FAMILIES
    ]).with_columns(pl.sum_horizontal(pl.all()).alias("blk_count"))
    ctx = pl.DataFrame({
        "is_s2": b["is_s2_2"], "m_nonascii": b["had_nonascii_2"],
        "m_addr_empty": b["addr_empty_2"], "s1_addr_empty": a["addr_empty_1"],
        "s1_core_tok": a["n_core_tok_1"], "m_core_tok": b["n_core_tok_2"],
        "s1_same_core": a["s1_same_core_1"], "m_same_core": b["s1_same_core_2"],
        "s1_same_addr": a["s1_same_addr_1"], "m_same_addr": b["s1_same_addr_2"],
        "s1_core_support": a["core_support_1"], "m_core_support": b["core_support_2"],
        "s1_name_digits": a["name_digits_1"], "m_name_digits": b["name_digits_2"],
    })
    return pl.concat([p, feats, nums, legal, bits, ctx], how="horizontal")
