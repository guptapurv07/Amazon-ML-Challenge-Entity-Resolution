"""Text normalization for business names and addresses.

Everything is country-agnostic: the same rules run on every country, and the
only learned component (the transliteration token dictionary) is mined from
matched training pairs.

Output columns added by `normalize_records` (+ `apply_learned_maps` for name_core):
  name_norm   : canonical full name (legal suffixes canonicalized, e.g. "pvt ltd")
  name_core   : name_norm without legal suffixes / connector words
  addr_norm   : canonical address (street-type abbreviations canonicalized)
  addr_nums   : list of digit sequences in the address (house / unit numbers)
  had_nonascii: whether the raw name contained non-ASCII characters
"""
import machine_limits  # noqa: F401  -- must stay the first import (caps CPU threads)
import polars as pl
from anyascii import anyascii

# Token-level canonical forms. Keys are post-lowercase, post-punctuation tokens.
NAME_CANON = {
    "private": "pvt", "pvt": "pvt", "pte": "pvt",
    "limited": "ltd", "ltd": "ltd", "limted": "ltd",
    "incorporated": "inc", "incorporation": "inc", "inc": "inc",
    "corporation": "corp", "corp": "corp", "corpn": "corp",
    "company": "co", "co": "co", "compagnie": "cie", "cie": "cie",
    "llc": "llc", "llp": "llp", "lp": "lp", "plc": "plc", "pllc": "pllc",
    "and": "and", "et": "and", "n": "and",
    "services": "services", "service": "services", "svcs": "services",
    "intl": "international", "international": "international",
    "technologies": "technology", "technology": "technology", "tech": "tech",
    "sarl": "sarl", "sas": "sas", "sasu": "sasu", "eurl": "eurl", "sa": "sa",
    "sci": "sci", "snc": "snc", "scop": "scop", "ste": "ste", "societe": "societe",
}

# Honorifics that the noisy sources prefix to names ("Sri Galaxy Builders",
# "Smt noidaservice.com", "Dr Age (India)").
HONORIFICS = {"sri", "shri", "smt", "dr", "mr", "mrs"}

# Tokens dropped from the core name: legal forms, connectors and honorifics.
LEGAL_TOKENS = {
    "pvt", "ltd", "inc", "corp", "co", "cie", "llc", "llp", "lp", "plc", "pllc", "pc",
    "and", "the", "of", "sarl", "sas", "sasu", "eurl", "sa", "sci", "snc", "scop",
    "dba", "aka", "fka", "www", "com", "net", "org", "in", "fr",
} | HONORIFICS

ADDR_CANON = {
    "street": "st", "st": "st", "str": "st",
    "road": "rd", "rd": "rd",
    "avenue": "ave", "ave": "ave", "av": "ave", "avn": "ave",
    "boulevard": "blvd", "blvd": "blvd", "bd": "blvd", "boul": "blvd",
    "drive": "dr", "dr": "dr",
    "court": "ct", "ct": "ct",
    "lane": "ln", "ln": "ln",
    "place": "pl", "pl": "pl",
    "circle": "cir", "cir": "cir",
    "highway": "hwy", "hwy": "hwy",
    "parkway": "pkwy", "pkwy": "pkwy",
    "terrace": "ter", "ter": "ter",
    "square": "sq", "sq": "sq",
    "suite": "ste", "ste": "ste",
    "apartment": "apt", "apt": "apt", "appt": "apt",
    "floor": "fl", "fl": "fl", "flr": "fl",
    "building": "bldg", "bldg": "bldg",
    "north": "n", "south": "s", "east": "e", "west": "w",
    "number": "no", "no": "no", "num": "no", "hno": "no", "h": "h",
    "route": "rte", "rte": "rte",
    "chemin": "chem", "chem": "chem",
    "impasse": "imp", "imp": "imp",
    "allee": "all",
    "rue": "rue", "r": "rue",
    "saint": "st", "sainte": "ste",
    "sector": "sec", "sec": "sec",
    "nagar": "ngr", "ngr": "ngr",
    "colony": "cly", "cly": "cly",
    "near": "near", "nr": "near", "opp": "opp", "opposite": "opp",
}


def _translit(s):
    return anyascii(s)


def transliterate(col):
    """Expression: map non-ASCII text (accents, Indic scripts) to ASCII."""
    return (
        pl.when(pl.col(col).str.contains(r"[^\x00-\x7F]"))
        .then(pl.col(col).map_elements(_translit, return_dtype=pl.String))
        .otherwise(pl.col(col))
    )


def basic_clean(expr, join_dots):
    """Lowercase, unify '&', strip punctuation, collapse spaces."""
    e = expr.str.to_lowercase().str.replace_all("&", " and ").str.replace_all("['`]", "")
    # "L.L.P." -> "llp" in names; "No.74" -> "no 74" in addresses
    e = e.str.replace_all(r"\.", "" if join_dots else " ")
    e = e.str.replace_all(r"[^a-z0-9 ]+", " ")
    return e.str.replace_all(r"\s+", " ").str.strip_chars()


LEET = {"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "6": "g", "7": "t", "8": "b"}


def fix_leet(expr_tokens):
    """In tokens mixing letters and digits ("br0wn", "materia1s"), map digits to
    look-alike letters. Pure-number tokens are left alone."""
    fixed = pl.element()
    for d, c in LEET.items():
        fixed = fixed.str.replace_all(d, c, literal=True)
    return expr_tokens.list.eval(
        pl.when(pl.element().str.contains(r"[a-z]") & pl.element().str.contains(r"\d"))
        .then(fixed).otherwise(pl.element()))


def canon_tokens(expr, mapping):
    """Map each whitespace token through `mapping` (unknown tokens unchanged)."""
    return expr.str.split(" ").list.eval(
        pl.element().replace(list(mapping.keys()), list(mapping.values()))
    )


def normalize_records(df):
    """Base normalization. Adds had_nonascii, name_norm, addr_norm, addr_nums
    and drops the raw text columns (join back on entity_id when needed).
    Call `apply_name_dict` afterwards to add the learned mapping and name_core.
    """
    df = df.with_columns(
        pl.col("entity_id").str.slice(0, 2).alias("src"),
        pl.col("business_name").str.contains(r"[^\x00-\x7F]").alias("had_nonascii"),
        transliterate("business_name")
        # "visioncenter.com" / "www.x.in" -> "visioncenter" / "x"
        .str.replace_all(r"(?i)^www\.", "")
        .str.replace_all(r"(?i)\.(com|net|org|biz|info|co\.in|in|fr|us)\b", " ")
        .alias("_name"),
        transliterate("business_address").alias("_addr"),
    )
    name_tokens = fix_leet(basic_clean(pl.col("_name"), join_dots=True).str.split(" "))
    name_tokens = name_tokens.list.eval(
        pl.element().replace(list(NAME_CANON.keys()), list(NAME_CANON.values())))
    # drop phone-like numbers ("AQX Investment - 3798245128") and repeated words
    # ("Atlantic Defense Defense Associates")
    name_tokens = name_tokens.list.eval(
        pl.element().filter(~pl.element().str.contains(r"^\d{6,}$")))
    name_tokens = name_tokens.list.eval(
        pl.element().filter((pl.element() != pl.element().shift(1)).fill_null(True)))
    addr_tokens = canon_tokens(basic_clean(pl.col("_addr"), join_dots=False), ADDR_CANON)
    return df.with_columns(
        name_tokens.list.join(" ").str.strip_chars().alias("name_norm"),
        addr_tokens.list.join(" ").str.strip_chars().alias("addr_norm"),
        pl.col("_addr").str.extract_all(r"\d+").alias("addr_nums"),
    ).drop("_name", "_addr", "business_name", "business_address")


def map_tokens(expr, mapping):
    """Map each space-separated token of a string column through `mapping`."""
    if not mapping:
        return expr
    return (expr.str.split(" ")
            .list.eval(pl.element().replace(list(mapping.keys()), list(mapping.values())))
            .list.join(" "))


def apply_learned_maps(df, maps):
    """Apply the learned token maps, then derive name_core (legal forms,
    connectors and honorifics removed).

    maps["script"]: word-by-word transliteration map, applied only to names that
                    were written in a non-Latin script;
    maps["name"], maps["addr"]: applied to every record, S1 included, so both
                    sides of a pair stay consistent.
    """
    name = map_tokens(pl.col("name_norm"), maps.get("script"))
    name = pl.when(pl.col("had_nonascii")).then(name).otherwise(pl.col("name_norm"))
    df = df.with_columns(map_tokens(name, maps.get("name")).alias("name_norm"),
                         map_tokens(pl.col("addr_norm"), maps.get("addr")).alias("addr_norm"))
    return df.with_columns(
        pl.col("name_norm").str.split(" ")
        .list.eval(pl.element().filter(
            ~pl.element().is_in(list(LEGAL_TOKENS)) & (pl.element().str.len_chars() > 0)))
        .list.join(" ").alias("name_core"),
    )


def one_word_substitutions(pairs):
    """For pairs (columns A = S1 text, B = S2/S3 text, plus extra columns) whose
    token sets differ by exactly one word on each side, return (a, b, extras...)."""
    extra = [c for c in pairs.columns if c not in ("A", "B")]
    d = pairs.with_columns(pl.col("A").str.split(" ").list.unique(),
                           pl.col("B").str.split(" ").list.unique())
    d = d.with_columns(pl.col("A").list.set_difference("B").alias("a"),
                       pl.col("B").list.set_difference("A").alias("b"))
    d = d.filter((pl.col("a").list.len() == 1) & (pl.col("b").list.len() == 1))
    return d.select(pl.col("a").list.first(), pl.col("b").list.first(), *extra)


def dominant_substitutions(subs, min_count, min_share):
    """Aggregate (a, b, nonascii) rows: for each b its most frequent a, with the
    count, the share among all substitutions of b, and the non-ASCII share."""
    cnt = subs.group_by("b", "a").agg(pl.len().alias("n"),
                                      pl.col("nonascii").mean().alias("nonascii"))
    cnt = cnt.with_columns((pl.col("n") / pl.col("n").sum().over("b")).alias("share"))
    return (cnt.sort("n", descending=True).group_by("b").first()
               .filter((pl.col("n") >= min_count) & (pl.col("share") >= min_share)
                       & (pl.col("a") != pl.col("b")) & (pl.col("a") != "")
                       & (pl.col("b") != "")))


def _no_chains(best):
    m = dict(zip(best["b"].to_list(), best["a"].to_list()))
    return {b: a for b, a in m.items() if a not in m}


def mine_token_maps(norm, gt, chunk=1_000_000):
    """Learn token maps from matched training pairs that differ by exactly one word.

    names: b -> a when b never occurs in any S1 name (a typo, transliteration or
           variant spelling), the substitution is frequent and dominant, and
           either the spellings are similar ("privte" -> "pvt", "lndia" -> "india")
           or the record was written in a non-Latin script ("siti" -> "city").
    addresses: frequent, near-unanimous substitutions without digits, e.g. state
           spellings ("mh", "mharastr" -> "maharashtra"; "texas" -> "tx").
    Direction is always S2/S3 spelling -> S1 spelling. Returns (name_map, addr_map).
    """
    from rapidfuzz import fuzz
    from rapidfuzz.process import cpdist

    s1 = norm.filter(pl.col("src") == "S1").select(
        pl.col("entity_id").alias("s1_id"), pl.col("name_norm").alias("an"),
        pl.col("addr_norm").alias("aa"))
    m = norm.filter(pl.col("src") != "S1").select(
        pl.col("entity_id").alias("match_id"), pl.col("name_norm").alias("bn"),
        pl.col("addr_norm").alias("ba"), pl.col("had_nonascii").alias("nonascii"))
    names, addrs = [], []
    for start in range(0, gt.height, chunk):
        p = gt.slice(start, chunk).join(m, on="match_id").join(s1, on="s1_id")
        names.append(one_word_substitutions(
            p.select(pl.col("an").alias("A"), pl.col("bn").alias("B"), "nonascii")))
        addrs.append(one_word_substitutions(
            p.select(pl.col("aa").alias("A"), pl.col("ba").alias("B"), "nonascii")))
    s1_vocab = s1.select(pl.col("an").str.split(" ").explode().unique())["an"]

    best = dominant_substitutions(pl.concat(names), min_count=3, min_share=0.5)
    best = best.filter(~pl.col("b").is_in(s1_vocab.implode()))
    sim = cpdist(best["a"].to_list(), best["b"].to_list(), scorer=fuzz.ratio, workers=4)
    best = best.with_columns(pl.Series("sim", sim)).filter(
        (pl.col("sim") >= 60) | (pl.col("nonascii") >= 0.5))
    name_map = _no_chains(best)

    best = dominant_substitutions(pl.concat(addrs), min_count=10, min_share=0.9).filter(
        ~pl.col("a").str.contains(r"\d") & ~pl.col("b").str.contains(r"\d"))
    addr_map = _no_chains(best)
    return name_map, addr_map


def mine_name_dict(norm, gt, min_count=3, min_share=0.5):
    """Learn {transliterated_token: latin_token} from matched training pairs.

    For each true pair whose S2/S3 name was non-ASCII and whose token count
    equals the S1 name's, tokens are aligned by position; a mapping is kept when
    it is frequent and dominant for that transliterated token.
    """
    s1 = norm.filter(pl.col("src") == "S1").select(
        pl.col("entity_id").alias("s1_id"), pl.col("name_norm").alias("a"))
    s23 = norm.filter(pl.col("had_nonascii") & (pl.col("src") != "S1")).select(
        pl.col("entity_id").alias("match_id"), pl.col("name_norm").alias("b"))
    pairs = (
        gt.join(s23, on="match_id").join(s1, on="s1_id")
        .select(pl.col("a").str.split(" "), pl.col("b").str.split(" "))
        .filter(pl.col("a").list.len() == pl.col("b").list.len())
        .explode("a", "b")
    )
    counts = pairs.group_by("b", "a").len("n")
    counts = counts.with_columns(
        (pl.col("n") / pl.col("n").sum().over("b")).alias("share"))
    best = (
        counts.sort("n", descending=True).group_by("b").first()
        .filter((pl.col("n") >= min_count) & (pl.col("share") >= min_share)
                & (pl.col("a") != pl.col("b")))
    )
    return dict(zip(best["b"].to_list(), best["a"].to_list()))
