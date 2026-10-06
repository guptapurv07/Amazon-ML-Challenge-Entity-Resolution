"""Learned cheap score for the one-to-one cut (replaces the hand-weighted BASE).

The cut keeps each S2/S3 record under its best S1 plus close runners-up, so the
score only has to rank the S1 candidates of a record well. Here that score is a
small gradient-boosted model on what is already known for every raw pair: the
cheap similarities (CHEAP_COLS) and which blocking families produced the pair.
Label = true pair (train ground truth). It is cross-fitted: S1 entities with
even / odd rid are scored by the model fitted on the other half, so train
candidates never get a score from a model that saw their own labels; test pairs
get the mean logit of both models.

For every train country the script reports pair completeness (PC) and
candidates per S1 of the current rule (BASE, m_max 3, delta 10) and of the
learned score with several (m_max, delta) rules, picks the smallest candidate
set within PC_SLACK of the best PC, and saves work/cut_model.joblib, which
candidates.py then uses for every split.

Run from student_resource/ after candidates.py --split train (needs the scored
raw pairs in work/scored/train/):  python src/cut_model.py
"""
import machine_limits  # noqa: F401  -- must stay the first import (caps CPU threads)
import joblib
import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingClassifier

from blocking import FAMILIES, PAIR_BASE
from candidates import (BASE, CHEAP_COLS, CUT, CUT_MODEL, cut_mask, cut_matrix, group_ranks,
                        load_records, load_scored)
from data_io import list_countries, load_gt_pairs, log
from model import SEED

CUT_FEATURES = CHEAP_COLS + [f"blk_{f}" for f in FAMILIES]
FIT_ROWS = 1_500_000  # sampled raw pairs per country and half
RULES = [(2, 1.0), (3, 1.5), (3, 2.0), (3, 2.5)]
PC_SLACK = 0.0005


def labels(raw, country, gt):
    ids = load_records("train", country).select("rid", "entity_id")
    t = (gt.join(ids.rename({"rid": "s1_rid", "entity_id": "s1_id"}), on="s1_id")
           .join(ids.rename({"rid": "m_rid", "entity_id": "match_id"}), on="match_id"))
    tk = np.sort(t["s1_rid"].cast(pl.UInt64).to_numpy() * PAIR_BASE + t["m_rid"].cast(pl.UInt64).to_numpy())
    rk = raw["s1_rid"].cast(pl.UInt64).to_numpy() * PAIR_BASE + raw["m_rid"].cast(pl.UInt64).to_numpy()
    pos = np.searchsorted(tk, rk).clip(0, len(tk) - 1)
    return tk[pos] == rk, len(tk)


def make_cut_model():
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, max_leaf_nodes=31,
                                          min_samples_leaf=200, early_stopping=True,
                                          validation_fraction=0.1, n_iter_no_change=20,
                                          random_state=SEED)


def main():
    gt = load_gt_pairs()
    data = {}
    Xs, ys, halves = [], [], []
    rng = np.random.default_rng(SEED)
    for country in list_countries("train"):
        raw = load_scored("train", country)
        y, n_true = labels(raw, country, gt)
        half = (raw["s1_rid"].to_numpy() % 2).astype(np.int8)
        idx = rng.choice(raw.height, min(raw.height, 2 * FIT_ROWS), replace=False)
        Xs.append(cut_matrix(raw[idx]))
        ys.append(y[idx])
        halves.append(half[idx])
        data[country] = n_true
        log(f"[{country}] {raw.height} raw pairs, {y.sum()} true ({y.mean():.3%}), "
            f"PC of raw blocking {y.sum() / n_true:.4f}")
        del raw, y, half
    X, y, half = np.concatenate(Xs), np.concatenate(ys), np.concatenate(halves)
    del Xs, ys, halves
    models = [make_cut_model().fit(X[half != h], y[half != h]) for h in (0, 1)]
    log(f"cut models fitted on {len(y)} sampled raw pairs ({[m.n_iter_ for m in models]} trees)")
    del X, y, half

    results = {}
    for country, n_true in data.items():
        raw = load_scored("train", country)
        y, _ = labels(raw, country, gt)
        s1, m = raw["s1_rid"].to_numpy(), raw["m_rid"].to_numpy()
        n_s1 = load_records("train", country).filter(pl.col("src") == "S1").height
        half = (s1 % 2).astype(np.int8)
        score = np.empty(raw.height, np.float32)
        for start in range(0, raw.height, 5_000_000):
            sl = slice(start, start + 5_000_000)
            Xc = cut_matrix(raw[sl])
            for h in (0, 1):  # a pair of the S1 half h is scored by the model fitted on the other half
                sel = half[sl] == h
                if sel.any():  # pairs are stored by S1 chunk, so a slice may hold one half only
                    score[sl][sel] = models[h].decision_function(Xc[sel])
        base = raw.select(BASE).to_series().to_numpy()

        def evaluate(sc, ranks, m_max, delta):
            rank, gid, _, top1, _ = ranks
            keep = cut_mask(sc, rank, top1[gid], None, s1, m_max, delta)
            return (y & keep).sum() / n_true, keep.sum() / n_s1

        pc, cps = evaluate(base, group_ranks(m, base), CUT["m_max"], CUT["delta"])
        ranks = group_ranks(m, score)
        log(f"[{country}] current rule (hand-weighted score, m_max 3, delta 10): "
            f"PC {pc:.4f} at {cps:.2f} cand/S1")
        for m_max, delta in RULES:
            pc, cps = evaluate(score, ranks, m_max, delta)
            results.setdefault((m_max, delta), []).append((pc, cps, n_s1, n_true))
            log(f"[{country}] learned score, m_max {m_max}, delta {delta}: PC {pc:.4f} at {cps:.2f} cand/S1")
        del raw, y, s1, m, score, base

    def total(rows):  # pooled over countries
        return (sum(pc * nt for pc, _, _, nt in rows) / sum(nt for *_, nt in rows),
                sum(c * ns for _, c, ns, _ in rows) / sum(ns for _, _, ns, _ in rows))
    pooled = {rule: total(rows) for rule, rows in results.items()}
    best_pc = max(pc for pc, _ in pooled.values())
    rule = min((r for r, (pc, _) in pooled.items() if pc >= best_pc - PC_SLACK), key=lambda r: pooled[r][1])
    log(f"chosen rule: m_max {rule[0]}, delta {rule[1]} -> pooled PC {pooled[rule][0]:.4f} "
        f"at {pooled[rule][1]:.2f} cand/S1 (best PC {best_pc:.4f})")
    joblib.dump({"models": models, "features": CUT_FEATURES,
                 "rule": {"m_max": rule[0], "delta": rule[1]}}, CUT_MODEL)
    log(f"saved {CUT_MODEL.name}")


if __name__ == "__main__":
    main()
