"""Learned "noise or decoy?" scores for the words that differ between two names.

Noisy sources insert or drop words ("Sri", "Services", "Center", "Pvt"), while a
decoy business differs from an S1 entity by a word too ("Exports", "Overseas",
"LLP"). Which differences are harmless is learned from the training labels:
for every word, how often a pair in which that word is EXTRA (only in the S2/S3
name) or MISSING (only in the S1 name) is a true match, as smoothed log-odds
relative to the overall match rate (target encoding). Training rows get
out-of-fold values, so a row never sees its own label; test rows use
statistics from all training rows. Unseen words score 0 (neutral), so the
features stay usable on countries without training data.
"""
import machine_limits  # noqa: F401  -- must stay the first import (caps CPU threads)
import numpy as np
import polars as pl
from rapidfuzz import fuzz, process
from rapidfuzz.distance import Levenshtein

SIDES = ("extra", "missing")
SMOOTH = 20.0  # pseudo-count pulling rare words towards neutral
TYPO_RATIO = 80  # an extra and a missing word this similar are one misspelled word


def name_diffs(pairs, rec):
    """List columns `extra` (words only in the S2/S3 name) and `missing` (only in
    S1), plus `n_typo`. An extra and a missing word that are nearly the same
    string ("privte"/"private", "sportif"/"sportive") are one misspelled word, not
    an inserted and a dropped one: they are removed from both lists and counted
    in `n_typo`. The word scores then describe real insertions and deletions
    only, so an unseen word (e.g. French) is judged as a new word, not as the
    typical unseen word of training (a typo)."""
    a = rec["name_norm"].gather(pairs["s1_rid"]).str.split(" ").list.unique()
    b = rec["name_norm"].gather(pairs["m_rid"]).str.split(" ").list.unique()
    nonempty = pl.element().filter(pl.element() != "")
    d = pl.DataFrame({"a": a, "b": b}).select(
        pl.int_range(pl.len(), dtype=pl.UInt32).alias("i"),
        pl.col("b").list.set_difference("a").list.eval(nonempty).alias("extra"),
        pl.col("a").list.set_difference("b").list.eval(nonempty).alias("missing"))
    x = (d.select("i", "extra").explode("extra", empty_as_null=True).drop_nulls()
          .join(d.select("i", "missing").explode("missing", empty_as_null=True).drop_nulls(), on="i"))
    if x.height:
        e, m = x["extra"].to_list(), x["missing"].to_list()
        ratio = process.cpdist(e, m, scorer=fuzz.ratio, workers=machine_limits.N_THREADS)
        edits = process.cpdist(e, m, scorer=Levenshtein.distance, workers=machine_limits.N_THREADS)
        shorter = np.minimum(x["extra"].str.len_chars().to_numpy(), x["missing"].str.len_chars().to_numpy())
        x = x.filter(pl.Series((ratio >= TYPO_RATIO) | ((edits <= 1) & (shorter >= 3))))
    typos = x.group_by("i").agg(pl.col("extra").unique().alias("t_extra"),
                                pl.col("missing").unique().alias("t_missing"))
    no_words = pl.lit([], dtype=pl.List(pl.String))
    return (d.join(typos, on="i", how="left")
             .with_columns(pl.col("t_extra").fill_null(no_words), pl.col("t_missing").fill_null(no_words))
             .sort("i")
             .select(pl.col("extra").list.set_difference("t_extra").alias("extra"),
                     pl.col("missing").list.set_difference("t_missing").alias("missing"),
                     pl.col("t_extra").list.len().cast(pl.UInt8).alias("n_typo")))


def fit_stats(diffs, label):
    """Per side and word: number of pairs (n) and of true pairs (pos)."""
    y = pl.Series("y", np.asarray(label, dtype=np.int64))
    out = {}
    for side in SIDES:
        e = (diffs.select(side).with_columns(y).explode(side)
                  .filter(pl.col(side).is_not_null() & (pl.col(side) != "")))
        out[side] = e.group_by(side).agg(pl.len().alias("n"), pl.col("y").sum().alias("pos")) \
                     .rename({side: "w"})
    return out


def subtract_stats(total, part):
    """total - part, per side (statistics of all rows except `part`)."""
    out = {}
    for side in SIDES:
        out[side] = (total[side].join(part[side], on="w", how="left", suffix="_p")
                     .with_columns(pl.col("n_p", "pos_p").fill_null(0))
                     .select("w", (pl.col("n") - pl.col("n_p")).alias("n"),
                             (pl.col("pos") - pl.col("pos_p")).alias("pos")))
    return out


def te_features(diffs, stats, prior):
    """Per pair: min / mean / max log-odds of its extra and missing words, the
    number of such words, and how many were never seen in training."""
    base = np.log(prior / (1 - prior))
    out = pl.DataFrame({"i": np.arange(diffs.height, dtype=np.uint32)})
    for side in SIDES:
        e = (diffs.select(pl.int_range(pl.len(), dtype=pl.UInt32).alias("i"), side)
                  .explode(side).filter(pl.col(side).is_not_null() & (pl.col(side) != ""))
                  .join(stats[side], left_on=side, right_on="w", how="left")
                  .with_columns(pl.col("n", "pos").fill_null(0)))
        lo = ((pl.col("pos") + SMOOTH * prior)
              / (pl.col("n") - pl.col("pos") + SMOOTH * (1 - prior))).log() - base
        agg = e.with_columns(lo.alias("lo")).group_by("i").agg(
            pl.col("lo").min().alias(f"te_{side}_min"),
            pl.col("lo").mean().alias(f"te_{side}_mean"),
            pl.col("lo").max().alias(f"te_{side}_max"),
            pl.len().alias(f"n_{side}"),
            (pl.col("n") == 0).sum().alias(f"te_{side}_unseen"),
        )
        out = out.join(agg, on="i", how="left")
    return (out.sort("i").drop("i")
               .with_columns(pl.col("n_extra", "n_missing", "te_extra_unseen", "te_missing_unseen")
                             .fill_null(0)))


def oof_te_features(diffs, label, fold_of, n_folds):
    """Out-of-fold target-encoding features for training rows, plus the
    statistics over all rows (for scoring new data)."""
    label = np.asarray(label)
    prior = float(label.mean())
    total = fit_stats(diffs, label)
    frames, order = [], []
    for f in range(n_folds):
        idx = np.flatnonzero(fold_of == f)
        part = fit_stats(diffs[idx], label[idx])
        frames.append(te_features(diffs[idx], subtract_stats(total, part), prior))
        order.append(idx)
    feats = pl.concat(frames)
    inverse = np.empty(len(label), dtype=np.int64)
    inverse[np.concatenate(order)] = np.arange(len(label))
    return feats[inverse], total, prior


PSEUDO_LO, PSEUDO_HI = 0.03, 0.97  # confident enough to act as labels


def pseudo_te_features(diffs, p, fold_of, n_folds, lo=PSEUDO_LO, hi=PSEUDO_HI):
    """Word scores for a country WITHOUT labels, learned from its own confident pairs.

    A model that does not use word scores rates every pair (p); pairs with p <= lo
    count as non-matches and p >= hi as matches. Their extra / missing words give the
    statistics, out-of-fold as in oof_te_features (fold_of groups pairs by S1), so
    the country's own decoy words (e.g. French ones) get scores instead of looking unseen.
    """
    conf = (p <= lo) | (p >= hi)
    label = (p >= hi).astype(np.int64)
    prior = float(label[conf].mean())
    total = fit_stats(diffs.filter(pl.Series(conf)), label[conf])
    frames, order = [], []
    for f in range(n_folds):
        idx = np.flatnonzero(fold_of == f)
        part = np.flatnonzero((fold_of == f) & conf)
        frames.append(te_features(diffs[idx], subtract_stats(total, fit_stats(diffs[part], label[part])), prior))
        order.append(idx)
    feats = pl.concat(frames)
    inverse = np.empty(len(p), dtype=np.int64)
    inverse[np.concatenate(order)] = np.arange(len(p))
    return feats[inverse]
