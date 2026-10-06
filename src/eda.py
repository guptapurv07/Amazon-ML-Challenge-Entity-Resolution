"""Exploratory data analysis. Prints a report and saves it to work/eda_report.txt.

Run from student_resource/:  python src/eda.py
"""
import machine_limits  # noqa: F401  -- must stay the first import (caps CPU threads)
import sys

import polars as pl

from data_io import WORK, read_ground_truth, read_sources

pl.Config.set_tbl_rows(40)
pl.Config.set_tbl_cols(10)
pl.Config.set_fmt_str_lengths(90)
pl.Config.set_tbl_width_chars(250)


class Tee:
    def __init__(self, path):
        self.f = open(path, "w", encoding="utf-8")

    def __call__(self, *args):
        text = " ".join(str(a) for a in args)
        print(text)
        self.f.write(text + "\n")


def with_source(df):
    return df.with_columns(pl.col("entity_id").str.slice(0, 2).alias("src"))


def main():
    WORK.mkdir(exist_ok=True)
    out = Tee(WORK / "eda_report.txt")

    train = with_source(read_sources("train"))
    test = with_source(read_sources("test"))

    out("=== Row counts per split / source / country ===")
    for name, df in [("train", train), ("test", test)]:
        out(f"-- {name}")
        out(df.pivot(on="country", index="src", values="entity_id",
                     aggregate_function="len", sort_columns=True).sort("src"))

    out("\n=== Empty-field rate per split / source / country ===")
    for name, df in [("train", train), ("test", test)]:
        out(f"-- {name}")
        out(df.group_by("src", "country").agg(
            pl.len().alias("n"),
            (pl.col("business_name") == "").mean().round(4).alias("empty_name"),
            (pl.col("business_address") == "").mean().round(4).alias("empty_addr"),
            pl.col("business_name").str.contains(r"[ऀ-ॿ]").mean().round(4)
              .alias("devanagari_name"),
            pl.col("business_address").str.contains(r"\b\d{6}\b").mean().round(4)
              .alias("addr_has_6digit"),
            pl.col("business_address").str.contains(r"\b\d{5}\b").mean().round(4)
              .alias("addr_has_5digit"),
        ).sort("src", "country"))

    out("\n=== Ground truth ===")
    gt = read_ground_truth()
    s1 = train.filter(pl.col("src") == "S1").select("entity_id", "country")
    counts = gt.group_by("s1_id").len("n_match")
    s1c = (s1.join(counts, left_on="entity_id", right_on="s1_id", how="left")
             .with_columns(pl.col("n_match").fill_null(0)))
    out("S1 train entities:", s1.height, "| with >=1 match:", counts.height,
        "| total true pairs:", gt.height)
    out(s1c.group_by("country").agg(
        pl.len().alias("n_s1"),
        (pl.col("n_match") == 0).mean().round(4).alias("singleton_rate"),
        pl.col("n_match").mean().round(3).alias("avg_matches"),
        pl.col("n_match").filter(pl.col("n_match") > 0).mean().round(3)
          .alias("avg_matches_nonsingleton"),
        pl.col("n_match").max().alias("max_matches"),
    ).sort("country"))
    out("Overall singleton rate:", round((s1c["n_match"] == 0).mean(), 4))
    out("Distribution of matches per S1 (0..10+):")
    out(s1c.with_columns(pl.col("n_match").clip(0, 10)).group_by("n_match").len()
        .sort("n_match").with_columns((pl.col("len") / s1c.height).round(4).alias("share")))

    s23 = train.filter(pl.col("src") != "S1").select(
        pl.col("entity_id").alias("match_id"), pl.col("country").alias("m_country"), "src")
    gtc = (gt.join(s1.rename({"entity_id": "s1_id", "country": "s1_country"}), on="s1_id", how="left")
             .join(s23, on="match_id", how="left"))
    out("\nMatched ids missing from S2/S3 files:", gtc["m_country"].is_null().sum())
    out("S1 ids in GT missing from S1 file:", gtc["s1_country"].is_null().sum())
    out("Matches crossing countries:", (gtc["s1_country"] != gtc["m_country"]).sum())
    out("Matched pairs by source:", gtc.group_by("src").len().sort("src").rows())

    multi = gt.group_by("match_id").agg(pl.col("s1_id").n_unique().alias("n_s1"))
    out("S2/S3 ids appearing under >1 S1:", multi.filter(pl.col("n_s1") > 1).height,
        "of", multi.height)
    out("Duplicate (s1, match) rows in GT:", gt.height - gt.unique().height)

    n_s23 = train.filter(pl.col("src") != "S1").group_by("src", "country").len("n_total")
    n_matched = gtc.group_by("src", "m_country").agg(pl.col("match_id").n_unique().alias("n_matched"))
    out("\nShare of S2/S3 records that belong to some S1 (rest are distractors):")
    out(n_s23.join(n_matched, left_on=["src", "country"], right_on=["src", "m_country"], how="left")
        .with_columns((pl.col("n_matched") / pl.col("n_total")).round(4).alias("share_matched"))
        .sort("src", "country"))

    out("\nPer-S1 mix: how many matched S1 have matches in both S2 and S3:")
    mix = gtc.group_by("s1_id").agg(pl.col("src").n_unique().alias("n_src"))
    out(mix.group_by("n_src").len().sort("n_src"))

    out("\n=== Sample matched groups (by country) ===")
    full = train.select("entity_id", "business_name", "business_address")
    for country in sorted(s1["country"].unique().to_list()):
        ids = (s1c.filter((pl.col("country") == country) & (pl.col("n_match") >= 2))
                  .sample(3, seed=0)["entity_id"].to_list())
        for sid in ids:
            members = [sid] + gt.filter(pl.col("s1_id") == sid)["match_id"].to_list()
            out(f"-- {country} {sid}")
            out(full.filter(pl.col("entity_id").is_in(members))
                .sort(pl.col("entity_id").str.slice(0, 2)))

    out("\n=== Sample singleton S1 rows ===")
    out(train.join(s1c.filter(pl.col("n_match") == 0).select("entity_id"), on="entity_id")
        .sample(8, seed=0).select("entity_id", "business_name", "business_address", "country"))

    out("\n=== Sample French test rows ===")
    fr = test.filter(pl.col("country") == "France")
    for src in ["S1", "S2", "S3"]:
        out(f"-- {src}")
        out(fr.filter(pl.col("src") == src).sample(8, seed=0)
            .select("entity_id", "business_name", "business_address"))

    out("\nCountries per split:", {n: sorted(d["country"].unique().to_list())
                                   for n, d in [("train", train), ("test", test)]})
    out.f.close()


if __name__ == "__main__":
    sys.exit(main())
