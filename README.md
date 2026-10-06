# Business Entity Resolution — Amazon ML Challenge 2026

Our team's solution to the Amazon ML Challenge 2026, a 72-hour hackathon hosted on Unstop
(25–27 September 2026, 89,399 registered participants).

**Team The Mavericks finished at rank 190 with a macro F0.5 of 0.988708** and was shortlisted
among the Top 1,000 teams for the methodology review.

## The problem

Business records arrive from three independent data vendors. There is no shared ID; the only
fields are business name, address and country.

- **Source 1** is a clean, deduplicated reference list: one record per real business.
- **Sources 2 and 3** are noisy fragments: abbreviations, typos, missing address parts,
  reordered words, other scripts.

For every Source-1 business, find **all** Source-2 and Source-3 records that describe the same
real-world business. A business can have many matches, one, or none.

What makes it hard:

- **Scale.** The test set has 1.73 million reference businesses and 9.97 million records.
  Comparing everything with everything is not an option.
- **Look-alikes.** Chains share a name at different addresses; malls share an address across
  different businesses.
- **The metric.** Macro F0.5 per business: a wrong match costs about twice as much as a missed
  one, and a business with no true match scores 1 only if we predict nothing for it.
- **An unseen country.** Training data covers India and the US. The test set adds France, with
  no labels at all.
- **Candidate-set size counts.** The organizers also rank the blocking stage: fewer candidates
  per business is better.

## Results

| Measure | Value |
|---|---|
| Final team result (public leaderboard, two pipelines merged) | **0.988708**, rank 190 |
| This repository's pipeline alone (public leaderboard) | 0.98515 |
| Validation, macro F0.5 on 550,877 held-out train businesses (weighted like the test set) | 0.9873 |
| Candidates per test business (this pipeline / final merged file) | **6.60** / 6.63 |
| Reduction ratio on test (within country) | 0.99999829 |
| True pairs kept by blocking (train) | India 97.2%, US 97.9% |

### What this repository is, and is not

The team built **two independent pipelines** and merged their results.

- **This repository is pipeline A**, plus the merge step. On its own it scores 0.98515.
- **Pipeline B** was built by Kartikey Kashyap. Its code is not in this repository.
- The final 0.988708 comes from merging pipeline A's probabilities with pipeline B's matches
  (`src/team_merge.py`). Reproducing that number needs pipeline B's output file.
- **Not included:** the competition dataset (it belongs to the organizers), any file derived
  from it (outputs, intermediate files, trained models), and the organizers' validation
  script.

## How it works

```
dataset/*.tsv
 -> normalization + learned token maps               prepare.py, normalize.py
 -> 14 hashed blocking key families, per country      blocking.py                  67 raw pairs per business
 -> cheap similarities + learned one-to-one cut       candidates.py, cut_model.py  6.6 pairs per business
 -> pair model: gradient boosting, 76 features        features.py, te.py, train.py, predict.py
 -> uncertain pairs: cross-encoder (GPU)              colab/*.ipynb                ~6% of the pairs
 -> stage 2: pair model + cross-encoder               stage2.py
 -> label-shift correction, one-to-one decision       decide.py
 -> (optional) merge with a second pipeline           team_merge.py
```

1. **Normalize** (`normalize.py`, `prepare.py`). Lowercasing, accents, transliteration of
   Devanagari to ASCII, legal forms (`Pvt Ltd`, `LLP`, `SARL`), address abbreviations. Token
   substitutions are learned from matched training pairs.
2. **Block** (`blocking.py`). Fourteen families of hashed keys are unioned per country: rare
   name words and their pairs, name × house number, name × address word, glued names, sorted
   names, acronyms, pairs of rare address words. Every key has a block-size cap, so nothing is
   compared all-pairs.
3. **Cut** (`cut_model.py`, `candidates.py`). The training labels show that a Source-2/3 record
   never belongs to two Source-1 businesses. So each record is kept only under its best few
   businesses (at most 3), ranked by a small learned model on cheap similarities. This takes
   67 raw pairs per business down to 6.6 while keeping 97–98% of the true pairs. A plain
   "top 5 per business" of about the same size kept only 82–85% (measured on an earlier
   blocking version).
4. **Pair model** (`features.py`, `te.py`, `train.py`). scikit-learn
   `HistGradientBoostingClassifier` with 76 features: fuzzy name and address similarity,
   house-number overlap and shift, legal-form agreement, chain and shared-address counts, rank
   and margin against competing candidates, and learned "noise or decoy?" scores for the words
   by which two names differ (an extra `Services` is harmless; an extra `Exports` usually
   means another company).
5. **Cross-encoder** (`ce_export.py`, notebooks). A fine-tuned `xlm-roberta-base` reads the raw
   text of the ~6% of pairs the pair model is unsure about.
6. **Stage 2** (`stage2.py`). A small gradient-boosting model combines the pair model's
   probability, the cross-encoder's score and how each pair compares with its competitors.
   Validation F0.5 rises from 0.9838 to 0.9874.
7. **Decision** (`decide.py`). The test set holds more distractor records than train, so the
   probabilities are recalibrated per country with expectation-maximization (no labels
   needed). Each record is then kept only under its most probable business, with a threshold
   of 0.725 chosen to maximize macro F0.5 on validation.

### France: a country without labels

No country name appears anywhere in the code. "A country without training data" is simply a
test country that is absent from train, and it gets its own recipe:

- The recipe was chosen by **leave-one-country-out validation**: train on India, score the US,
  and the reverse.
- The word scores are re-learned from the country's own most confident predictions
  (self-training). In the leave-one-country-out test this scored 0.9638, against 0.9574
  without word scores and 0.9380 with word scores carried over from the training countries.
- The cross-encoder is fine-tuned further on that country's confident pairs.

### Team merge

`src/team_merge.py` treats the second pipeline's decision as evidence of fixed strength: a
pair's odds are multiplied by 8 if pipeline B matched it and divided by 8 if not. Where
pipeline A is confident it decides; near the threshold pipeline B decides. For countries
without training data, pipeline B's matches are used as they are. Merged, the two pipelines
scored higher than either one alone.

## Running it

```bash
# copy the challenge data into dataset/train/*.tsv and dataset/test/*.tsv first
python3.13 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python src/run_pipeline.py
```

The command is re-entrant. Each finished step leaves a marker in `work/.done/`; at the GPU
steps it stops and prints which notebook to run and which files to copy back; running it again
continues where it stopped.

| Phase | Where | What | Time |
|---|---|---|---|
| 1 | CPU | normalize, blocking, learned cut, pair model, test scores, cross-encoder files | ~3 h |
| 2 | GPU (Google Colab / Kaggle) | three cross-encoder rounds (`colab/*.ipynb`) | ~3 h on one T4 |
| 3 | CPU | stage 2, decision, optional team merge | ~2 min |

- Times were measured on an Apple M2 with 8 GB RAM, capped at 4 threads
  (`src/machine_limits.py`). Peak memory is about 8.7 GB, in `train.py`, using swap.
- Output: `output/matching_results.tsv` and `output/candidate_pairs.tsv`.
- To add the team merge: `--mate <the second pipeline's matching_results.tsv>`.
- If the organizers' `utils/validate_submission.py` is present, it runs at the end.
- Seeds are fixed, and samples and folds come from deterministic hashes of entity IDs. GPU
  fine-tuning is not bit-exact across runs.

## Models and rules

- **Models:** `FacebookAI/xlm-roberta-base` (MIT license, 278M parameters) and scikit-learn
  gradient boosting. `kaggle/cross_encoder_large.ipynb` (`xlm-roberta-large`, MIT, 560M) is an
  experiment that gave no gain in this pipeline.
- **Data:** only the files the organizers provided. No external data, APIs, geocoding or
  business registries, as the challenge rules require.
- **Labels:** no pair was labeled by hand. Pseudo-labels for France come from the pipeline's
  own model.
- **Countries:** never hard-coded, and there is no country feature.

## Repository layout

| Path | Purpose |
|---|---|
| `src/run_pipeline.py` | one-command pipeline |
| `src/normalize.py`, `prepare.py` | normalization, learned token maps |
| `src/blocking.py`, `candidates.py`, `cut_model.py` | blocking, cheap scores, learned one-to-one cut |
| `src/features.py`, `te.py`, `model.py`, `train.py`, `predict.py` | pair model, validation, test scores |
| `src/ce_export.py`, `colab/*.ipynb` | cross-encoder rounds 1–3 (GPU) |
| `src/stage2.py` | pair model + cross-encoder combination |
| `src/decide.py`, `metrics.py` | decision layer; macro F0.5 exactly as the leaderboard defines it |
| `src/team_merge.py` | merge with a second pipeline |
| `src/eda.py`, `blocking_eval.py`, `cut_eval.py`, `decision_eval.py`, `diagnose.py`, `selftrain_eval.py`, `france_vote.py`, `baseline0.py` | analysis scripts and experiments behind the design choices |
| [`METHODOLOGY.md`](METHODOLOGY.md) | the full methodology document we submitted: numbers, tables, error analysis |
| [`experiments.md`](experiments.md) | the working log: every experiment and every leaderboard upload, from 0.940 to 0.988708 |

## Team

The Mavericks, Graphic Era (Deemed to be University), Dehradun:

- **Apurv Gupta** — team lead; pipeline A (this repository)
- **Kartikey Kashyap** — pipeline B
- **Sarthak Singh**
- **Siddhi Dhingra**

Thanks to Amazon and Unstop for organizing the challenge.
