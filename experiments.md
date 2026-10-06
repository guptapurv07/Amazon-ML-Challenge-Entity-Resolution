# Experiment log

> **Note for readers of this repository.** This is our working log, kept during the 72 hours, with every experiment and every leaderboard upload. "The teammate's pipeline" is pipeline B, built by Kartikey Kashyap (see `METHODOLOGY.md`, section 6). Paths under `work/`, `output/` and `Friend output/` are local folders that hold competition data or files derived from it; they are not in this repository.

Scores are macro F0.5 unless noted. CV = GroupKFold by S1 on the sampled train S1
entities; LOCO = train on one country, score the other; LB = public leaderboard.

| Date | Change | CV F0.5 | LOCO F0.5 | LB | Notes |
|---|---|---|---|---|---|
| 2026-09-25 | Baseline 0: every S1 predicted singleton | 0.0558 (train singleton rate) | – | ? | validator PASS |
| 2026-09-25 | Blocking v1 (4 key families, strict caps) | – | – | – | PC India 0.800 / US 0.899, 18.6 / 15.9 cand per S1 |
| 2026-09-25 | Blocking v2 (S1-vocabulary rare tokens, name x street key, concatenated-name key, leet-digit fix) | – | – | – | PC India 0.927, oracle F0.5 0.971, 44.5 cand per S1 |
| 2026-09-25 | Candidate trim: top-15 per S1 by name_tsr + 0.5*addr_tsr | – | – | – | 13.3 (India) / 13.7 (US) cand per S1; PC after trim on 10% sample: India 0.894, US 0.937 |
| 2026-09-25 | Baseline 1: rules name_tsr>=85 & addr_tsr>=70, one-to-one | 0.7230 | – | – | same 10% S1 sample as the model |
| 2026-09-25 | Model v1: HistGB (400 trees, 63 leaves), 40 features, threshold 0.675, one-to-one | **0.9499** (India 0.9348, US 0.9599) | not run | **0.940** (rank ~625; top of LB 0.988) | 10% S1 sample (3.0M pairs), 4 hash-grouped folds; all folds hit 400 trees. Test: 23.7M cand pairs, 5.39M matches; empty rate France 5.8% / India 7.1% / US 6.2%; avg matches 3.18 / 3.03 / 3.18. Validator PASS |
| 2026-09-26 | **Problem-statement update:** smaller candidate set per S1 now ranks higher in the final evaluation (see the problem statement) | – | – | – | v1 had 13.7 cand per S1 |
| 2026-09-26 | Candidate cut v2 (`cut_eval.py`, full train): cheap score = max(name token-set, joined-name ratio) + 0.5 × address token-set; **one-to-one cut**: each S2/S3 record kept under its best S1, plus 2nd/3rd if within 10 points | – | – | – | India **5.36** cand/S1, PC 0.9126, oracle 0.9659 (v1: 13.34, 0.8949, 0.9605). US **5.05**, PC 0.9634, oracle 0.9884 (v1: 13.72, 0.9374, 0.9804). Per-S1 top-5 would give 4.9 cand/S1 but PC only 0.817 / 0.850. Score floors and extra per-S1 picks changed nothing |
| 2026-09-26 | Model v2: v2 candidates (5.2/S1), 52 features (+ legal-form same/partial/conflict, position-free house numbers, first-number similarity, decoy-vs-typo name support, digits in name), 25% S1 sample (2.85M pairs), HistGB 600 trees lr 0.1, threshold 0.675 | **0.9646** (India 0.9514, US 0.9734) | not run | **0.957** plain, **0.958** with prior shift | all 4 folds used 600 trees. Expected-F0.5 rule 0.9642 (not better). EM prior shift recovers train share (India 0.5885 vs 0.5912, US 0.6618 vs 0.6598). Test EM share: France 0.488, India 0.461, US 0.545 (train 0.631): more decoys in test. Test: 11.4M cand (6.6/S1), variant A 5.57M matches |
| 2026-09-26 | Data refinement + blocking v3: token maps learned from train pairs (682 Devanagari-word, 12,650 name-word, 1,815 address-word substitutions), honorifics as legal tokens, phone-like numbers and repeated words dropped from names; new keys: whole name x house number, whole name x any address word, initials x number (acronyms); cheap score + acronym match + 0.25 x house-number similarity; same one-to-one cut | – | – | – | Raw: India 55.2 / US 50.2 pairs per S1. After cut: India **5.41** cand/S1, PC **0.9559**, oracle **0.9863** (v2 5.36, 0.9126, 0.9659); US **5.01**, PC **0.9687**, oracle **0.9903** (v2 5.05, 0.9634, 0.9884). Number weight 0.5 or a within-15 rule: +0.0004 oracle for +0.3 cand/S1, not worth it |
| 2026-09-26 | Model v3 smoke test (2% S1 sample): + target-encoded name-word differences (is an extra/missing word noise or a decoy marker?), 67 features | 0.9771 (India 0.9747, US 0.9787); test mix 0.9765 | 0.893 (2% sample, word scores from the other country only) | – | only a pipeline check; full 25% run follows |
| 2026-09-26 | **Model v3**: blocking/data v3 candidates + 67 features (v2 + target-encoded name-word differences: noise word vs decoy marker), 25% S1 sample (2.85M pairs), HistGB 900 trees (all folds hit the cap), threshold 0.725, EM prior shift | **0.9794** (India 0.9777, US 0.9806); **test mix 0.9790** | India 0.9072, US 0.9215 at CV threshold (0.9259 / 0.9413 at threshold 0.95); word scores from the other country only | pending | LB estimate 0.9693 with LOCO as France (pessimistic: LOCO trains on one country); with France as implied for v2 (0.939): ~0.973. **LB 0.970**: the LOCO-based estimate was the accurate one. Test: 11.54M cand (France 6.89, India 7.00, US 6.16 per S1), 5.71M matches; EM true share France 0.517, India 0.470, US 0.547 (train 0.645); empty France 5.1%, India 6.0%, US 5.8%. Peak memory: train 7.55 GB (near the 8 GB limit), predict 6.1 GB. Validator PASS (`work/sub_v3/`) |
| 2026-09-26 | v3b smoke test (2% sample): typo-aware word differences (+ n_typo), threshold grid up to 0.99, LOCO compares a model WITH vs WITHOUT word scores for countries without training data | 0.9764; test mix 0.9757 | with word scores 0.887 / 0.870 (India / US held out), **without 0.922 / 0.960** at CV threshold; best: without, threshold 0.887 → mean 0.947 | – | word scores do not transfer to an unlabeled country (all its words are unseen); LB estimate 0.9611 → 0.9714 at 2%. Full run started 12:48 |
| 2026-09-26 | **Model v3b**: typo-aware word differences (+ n_typo), up to 1500 trees (folds used 919-1148), threshold grid to 0.99; countries without training data get a model WITHOUT word scores (chosen by LOCO) and threshold 0.875 | 0.9789 (India 0.9772, US 0.9800); test mix 0.9785 | with word scores 0.907 / 0.911 (India / US held out, CV threshold); **without 0.939 / 0.960**; chosen: without, threshold 0.875 → mean 0.9526 | not uploaded (team decision: wait for a bigger jump) | **LB estimate 0.9746** (v3's estimate 0.9693 matched its LB 0.970). Test: France 3.25 matches/S1, 5.9% empty (v3 3.35, 5.1%); EM share France 0.495 with its own model. Peak memory train 7.84 GB, predict 6.0 GB. `work/sub_v3b/`, validator PASS |
| 2026-09-26 | Cross-encoder export for Colab (`ce_export.py`): hard pairs = stage-1 p in [0.02, 0.98] | – | – | – | ce_train 357,771 pairs (30% of train S1 outside the model sample: all hard + 3% easy, positive rate 0.53), ce_oof 217,495 (7.6% of the sample), ce_test 1,128,420 (France 436K = 24% of its pairs, US 380K = 9%, India 312K = 5.5%). Stage 2 dry run with fake scores: pipeline OK; stage-1 AUC on hard pairs 0.915 |
| 2026-09-26 | Blocking v4: glued legal words stripped ("bababusinessprivate"), order-free core (cats, sn), every pair of the 3 rarest name tokens (tt) and x house number (ttn), pairs of house numbers (nn) | – | – | – | Raw: India 62.5 / US 54.3 per S1 (v3 55.2 / 50.2). After the cut: India PC **0.9595** at 5.53/S1 (v3 0.9559 at 5.41), US **0.9716** at 5.06/S1 (v3 0.9687 at 5.01). Missed by blocking India 3.53% → 3.01%, but lost at cut 0.88% → 1.05%. Unrecoverable-looking: empty address + common name (India 1.26%: median 49 S1 share the name) |
| 2026-09-26 | **Learned cut** (`cut_model.py`): the one-to-one cut ranks a record's S1 candidates with a small HistGB on the cheap similarities + blocking-family flags (label = true pair; cross-fitted by S1 rid parity on train; test gets the mean of both models) instead of name + 0.5 addr + 0.25 number | – | – | – | Rule chosen: m_max 3, delta 2.0 logit. India PC **0.9672 at 5.23/S1** (hand-weighted cut on the same blocking: 0.9595 at 5.53; v3 0.9559 at 5.41), US PC **0.9745 at 4.99/S1** (0.9716 at 5.06; v3 0.9687 at 5.01). m_max 2, delta 1: India 0.9649 at 4.85, US 0.9721 at 4.76 |
| 2026-09-26 | **Cross-encoder + stage 2 on v3b**: xlm-roberta-base (MIT, 278M) fine-tuned 1 epoch on 350,654 hard/easy pairs of train S1 outside the model sample (Colab T4, 43 min); stage 2 = small HistGB on logit(p1), CE logit, CE gap within the S1 and margin within the record, hard-pair counts; trained out-of-fold on the 217,495 hard pairs of the sample | **0.9834** (India 0.9819, US 0.9844); **test mix 0.9830** (v3b 0.9785) | France not measurable locally | **0.979** (France also stage 2) | Held-out check in Colab (hard pairs): AUC CE 0.960 vs pair model 0.923, log loss 0.254 vs 0.352. AUC on the sample's hard pairs: stage 1 0.915, CE 0.959, stage 2 **0.976**. Threshold 0.750. France variants: stage 1 only (`work/sub_v3b_ce`, LB est. 0.978) or stage 2 with z1 from its own model (`work/sub_v3b_ce_fr`: 3.31 matches/S1, 5.7% empty vs 3.25, 5.9%) |
| 2026-09-26 | Error analysis of v3b + CE (`work/staged/errors.py`, out-of-fold on the sample) | – | – | – | Remaining loss 0.0166 of macro F0.5. Worth if fixed: blocking/cut misses +0.0116, true pairs among the candidates but not matched +0.0039, false matches +0.0014. False matches: 1,735 in the hard zone, 1,169 with stage-1 p > 0.98 (never sent to the CE). A fill-empty rule for predicted-empty S1 entities lowered CV: dropped. Unrecoverable-looking: empty address + common name (India 1.26%, US 1.34%) |
| 2026-09-26 | **Model v4**: blocking v4 + learned-cut candidates, same features as v3b; folds used 901-1402 trees; France keeps v3b's rule (without word scores, threshold 0.875; LOCO not rerun) | **0.9811** (India 0.9805, US 0.9816); **test mix 0.9810** (v3b 0.9785) | not rerun | not uploaded | Test: 11.33M cand (France 7.01, India 6.71, US 6.14 per S1), 5.73M matches; empty France 5.9%, India 6.0%, US 5.8%. `work/sub_v4/` (validator PASS); model, OOF and test predictions saved in `work/v4/` |
| 2026-09-26 | **v4 + cross-encoder stage 2** (round-1 CE; top-up scores for 54,912 new OOF and 234,075 new test hard pairs) | **0.9856** (India 0.9853, US 0.9859); **test mix 0.9856** (v3b + CE 0.9830) | France not measurable locally | not uploaded (est. ~0.981) | AUC on the sample's hard pairs: stage 1 0.908, CE 0.951, stage 2 0.971. Threshold 0.725. France also stage 2 (248,196 hard pairs, z1 from its own model). 5.77M matches from 11.33M candidates. `work/sub_v4_ce_fr/` |
| 2026-09-26 | **Blocking/data v5**: + a single very rare name token (t1, caps 5/100), + pairs of the 3 rarest address tokens (aa), address "r" → "rue"; learned cut refit on the v5 raw pairs (rule m_max 3, delta 2.0) | – | – | – | Raw: India 68.2 / US 59.8 pairs per S1, raw PC India **0.9748** / US **0.9827** (v4 0.9699 / 0.9773). After the cut: India PC **0.9716 at 5.28/S1** (v4 0.9672 at 5.23), US **0.9785 at 5.05/S1** (v4 0.9745 at 4.99); pooled 0.9758 at 5.14 (best rule 0.9759). Test: 11.43M cand (France 6.99, India 6.78, US 6.22 per S1; v4 7.01 / 6.71 / 6.14). Model v5 not trained yet |
| 2026-09-26 | **Model v5**: v5 candidates, 74 features; LOCO compares 3 variants for countries without training data: with word scores / without / **self-trained word scores** (statistics from the held-out country's own confident pairs, rated by the model without word scores) | **0.9823** (India 0.9819, US 0.9825); **test mix 0.9822** (v4 0.9810) | at CV threshold, India / US held out: with 0.9096 / 0.9096, without 0.9381 / 0.9631, **self-trained 0.9504 / 0.9642**; mean: without 0.9547 (threshold 0.887), **self-trained 0.9573 (0.725, chosen)** | – | Folds used 1095-1500 trees. Test: 11.43M cand, 5.72M matches. Train 54 min, peak 8.66 GB (swap); predict 14 min, 5.7 GB. `work/sub_v5/` validator PASS; artifacts incl. loco.parquet in `work/v5/`. Round-2 CE export: ce2_train 510,000 pairs (450,000 train S1 buckets 55-99 + 60,000 French pairs pseudo-labeled by the v4 model at p <= 0.03 / >= 0.97), ce2_oof 217,895, ce2_test 1,121,801, ce2_oof_loco 224,195 |
| 2026-09-26 | Leave-one-country-out check of self-training the MODEL for countries without training data (`selftrain_eval.py`: refit the model without word scores on the other country + the held-out country's confident pairs, p <= 0.03 / >= 0.97) | – | India 0.9457 -> 0.9456, US 0.9648 -> 0.9675 (best thresholds); mean 0.9547 -> **0.9562** (self-trained word scores already give 0.9573) | – | Pseudo-label quality: India 0.9795 precision among rated true, 1.5% true among rated false; US 0.9933 / 1.8%. Small gain, not used |
| 2026-09-26 | **v5 + cross-encoder round 2** (xlm-roberta-base fine-tuned further on 499,801 pairs incl. 60,000 French pseudo-labeled ones; Colab held-out, same pairs: hard-pair AUC round 1 0.9480 -> round 2 0.9568, log loss 0.284 -> 0.257) | **0.9870** (India 0.9870, US 0.9871); **test mix 0.9870** | France not measurable locally (stage 2 on 221,267 hard pairs, z1 from the self-trained-word-score model) | pending | AUC on the sample's hard pairs: stage 1 0.907, CE2 0.957, stage 2 0.974. Threshold 0.725. 5.79M matches. Safety variant with round-1 scores (86% OOF / 73% test coverage of v5 hard pairs): 0.9864. `work/sub_v5_ce2/` validator PASS, copied to `output/` (md5 cb07c625..., 97,068,693 bytes); previous output backed up in `work/output_backups/lb0979_v3b_ce_fr/`. LB estimate ~0.984 |
| 2026-09-26 | Missed-pairs breakdown on v5 (`work/staged/misses.py`) | – | – | – | Kept: India 0.9716, US 0.9785. Largest remaining group: S2/S3 record with an EMPTY address whose core name ~27-50 S1 share: India 1.23% missed by blocking + 0.16% at the cut, US 1.31% + 0.24%. Then India: no name word shared but a number shared 0.56%, common name + number 0.42% |
| 2026-09-26 | **Model v6**: + `num_first_diff` (record's first address number minus the S1's) and `num_min_shift` (smallest nonzero difference over all number pairs): look-alike records a few doors away (tip from a teammate: house number +1..+21) | **0.9838** (India 0.9827, US 0.9845); **test mix 0.9835** (v5 0.9822) | at CV threshold, India / US held out: without 0.9435 / 0.9713, **self-trained 0.9576 / 0.9700**; mean self-trained **0.9638** (v5 0.9573), chosen at 0.725 | – | The features transfer across countries (France proxy +0.0065). Folds 1077-1205 trees. Train 51 min, peak 8.2 GB; predict 15 min. `work/sub_v6/` PASS, artifacts in `work/v6/` |
| 2026-09-26 | **v6 + CE round 2** (top-up scores for 95,259 new hard pairs) | 0.9874 (India 0.9870, US 0.9876); **test mix 0.9873** (v5 + CE2 0.9870) | France stage 2 chosen by leave-one-country-out: own stage 2 0.9804 vs seen countries' 0.9802 (tie; not used) | – | The stage-1 gain mostly overlaps with what the CE already fixed. Stage 2 with 11 extra pair features (`--pair-feats`): hard-pair AUC 0.957 -> 0.959, CV unchanged (0.9872): not used |
| 2026-09-26 | **CE round 3 for France** (`colab/cross_encoder_round3.ipynb`): model2 fine-tuned further on 150,000 French pairs pseudo-labeled by v5 (p_unseen <= 0.03 / >= 0.97) + 50,000 labeled train pairs; rescored France's 183,587 hard test pairs | – | not measurable locally | pending (upload #8) | Round 3 vs round 2 on France's hard pairs: correlation 0.56, share with logit > 0: 50% vs 71% (India/US hard pairs are 51% true). France matches per S1: round 2 3.340, round 3 3.279 (India 3.33, US 3.36; France has 5.53 S2/S3 records per S1 vs ~5.8). Averaging both rounds: 3.261. Upload #8 = v6 + CE2 (India/US) + CE3 (France): `work/sub_v6_final_ce3fr/`, PASS, in `output/` (md5 38402508..., 96,910,237 bytes); variant with round 2 for France in `work/sub_v6_final_r2fr/` |

## CV vs leaderboard (why local CV was optimistic)

The CV average is weighted like the training sample (India 40%, US 60% of S1), but the
test set is India 46.8%, US 38.3%, France 15.0% of S1 (259,452 / 809,986 / 663,106).
Reweighting the per-country CV to the test mix (India 55%, US 45% of India+US):

| Model | CV (sample mix) | CV (test mix) | LB | Implied France |
|---|---|---|---|---|
| v1 | 0.9499 | 0.9461 | 0.940 | ~0.905 |
| v2 (+ prior shift on LB) | 0.9646 | 0.9613 | 0.958 | ~0.939 |
| v3 (+ prior shift) | 0.9794 | 0.9790 | 0.970 | ~0.919 |
| v3b + cross-encoder stage 2 (France too) | 0.9834 | 0.9830 | 0.979 | ~0.956 |
| v5 + cross-encoder round 2 (French pseudo-labels; France self-trained word scores) | 0.9870 | 0.9870 | 0.98396 | ~0.967 |
| v6 + CE round 2, France CE round 3 | 0.9874 | 0.9873 | 0.98515 | ~0.973 |

"Implied France" = (LB - 0.85 x test-mix CV) / 0.15, assuming India/US score on test as in CV.
The residual also absorbs the extra decoys in test. From now on report CV in the test mix,
and estimate France with leave-one-country-out.

**v3 (2026-09-26):** LB 0.970 matches the LOCO-based estimate (0.9693), not the "France as for
v2" guess (0.973). Implied France fell from ~0.94 (v2) to ~0.92: the new word-difference
features do not transfer to a country without labels (French words are all unseen), as
LOCO warned (0.907 / 0.922). From now on the LOCO-based leaderboard estimate is the
number to beat, and France (15% of test) is the biggest single gap.

## Leaderboard submissions (limit: 5 per day, 3 days)

The portal counts days in **UTC** (resets at 05:30 IST). All 5 uploads below (25 Sept 23:45 to 26 Sept 02:40 IST) fell on 25 Sept UTC, and a 6th upload at ~02:45 IST was refused.

| # | Date (IST) | File | What it tests | Public LB |
|---|---|---|---|---|
| 1 | 2026-09-25 ~23:45 | v1 `output/matching_results.tsv` | first model | 0.940 |
| 2 | 2026-09-26 ~00:50 | v2 plain threshold (variant A) | cut v2 + 52 features + 25% sample | 0.957 |
| 3 | 2026-09-26 ~00:50 | v2 + EM prior shift (variant B) | test has more decoys than train | **0.958** |
| 4 | 2026-09-26 ~02:20 | `output/matching_results.tsv`, which still held v2 B (v3 was only in `work/sub_v3/`) | – (uploaded by mistake) | 0.958 (same as #3) |
| 5 | 2026-09-26 ~02:40 | v3 (`output/`, identical to `work/sub_v3/`) | blocking/data v3 + word-difference features | **0.970** |
| 6 | 2026-09-26 ~15:40 | v3b + cross-encoder stage 2, France also stage 2 (`output/`, 96,130,481 bytes, identical to `work/sub_v3b_ce_fr/`) | does the cross-encoder help France? (India/US CV +0.0045) | **0.979** |

| 7 | 2026-09-26 ~20:15 | v5 + cross-encoder round 2 (`output/`, 97,068,693 bytes, md5 cb07c625..., identical to `work/sub_v5_ce2/`) | v5 blocking + self-trained word scores for France + CE round 2 trained with French pseudo-labeled pairs (test-mix CV 0.9870, estimate ~0.984) | **0.98396** |
| 8 | 2026-09-26 ~22:05 | v6 + CE round 2 (India/US) + CE round 3 (France) (`output/`, 96,910,237 bytes, md5 38402508..., identical to `work/sub_v6_final_ce3fr/`) | house-number shift features + France cross-encoder trained on 150k French pseudo-labeled pairs (test-mix CV 0.9873, estimate 0.985-0.986; ~0.9845 if round 3 neutral) | **0.98515** |
| – | 2026-09-26 (teammate's upload) | teammate's pipeline (`Friend output/matching_results_ce.tsv`): 98-99% of pairs shared with #8; +4.7% French matches; 52,017 pairs outside our candidates (acronyms "fg", glued names "quirozcohenlithiumcom", typos) | our model-based estimate was 0.9736 (range 0.972-0.978): wrong by +0.012, our probabilities under-rate pairs our model disagrees with | **0.98542** |
| 9 | 2026-09-26 ~23:15 | team merge (`work/staged/merge_team.py --mate-unseen`, `output/`, 98,030,435 bytes, md5 cf76c6f4..., = `work/sub_team_merge_fr/`): countries without training data (France) = teammate's matches as they are; India/US = our stage-2 probabilities with his vote as evidence (odds x3 / /3) at our threshold; + his pairs outside our candidates (p 0.9, one-to-one) | combine both (estimate ~0.986, 0.985-0.987) | **0.98638** |

| 10 | 2026-09-26 23:18 | team merge with OUR France (his vote x3 as tie-breaker there too; `work/sub_team_merge/`, 97,689,069 bytes, md5 45e7b383...) | France A/B against #9 (only France differs: 3.330 vs 3.433 matches per S1) | 0.98633 (#9 0.98638: his France marginally better, essentially a tie) |
| 11 | 2026-09-26 ~23:30 | #9 + 7,961 of our French matches with q >= 0.97 that he rejected, records still free (`work/merge_fr_his_plus_our_sure/`, 98,132,764 bytes, md5 d77a8db9...; built ad hoc, not yet in src/) | union of both sides' confident French picks (France 3.463 per S1; estimate ~0.9865) | 0.986333 (uploaded twice by mistake, 23:34 and 23:37: same file) |

| – | 2026-09-27 00:13 | stage 2 with round-1 AND round-2 CE (`--ce-extra-prefix ce`) | CV only | hard-pair AUC 0.9602 -> 0.9607, test-mix CV 0.9873 -> 0.9873: no gain |
| – | 2026-09-27 00:14 | expected-F0.5 decision per S1 (`decide_expected_f`, extra 0 / 0.05 / 0.10) instead of the threshold, on stage-2 OOF | CV only | test mix 0.9871 vs 0.9873 with the threshold: worse |
| 12 | 2026-09-27 ~00:15 | #9 with his vote at x1.5 instead of x3 on India/US (`team_merge.py --mate-unseen --vote-lr 1.5`, 97,956,944 bytes, md5 fb61674b...; +2,046 / -7,758 India/US pairs vs #9) | LB-only: how strongly to count his India/US vote | 0.986093 (x3: 0.986375; his India/US alone, i.e. x infinity: 0.985417) |
| 13 | 2026-09-27 ~00:25 | vote x8 on India/US (`--vote-lr 8`, 98,076,606 bytes, md5 508d340f...; +5,932 / -2,344 India/US pairs vs x3) | a quadratic fit in log(x) through x1.5 / x3 / x-infinity peaks near x20 at ~0.9867 | **0.986435** (best; the curve is flat near x3-x8) |
| 14 | 2026-09-27 ~00:28 | x8 + the teammate's NEW file (`Friend output/matching_results_cel_fr.tsv`, md5 a98f8026...: xlm-roberta-large CE + France rule chosen by leave-one-country-out; his hold-out 0.9902) with his France as is (`work/merge_cel_fr_x8/`, 97,910,983 bytes, md5 14353b9b...) | France: 877,132 pairs instead of 890,613 (-17,409 / +3,928); India/US +-1-2k | lower than #13 (exact score not recorded): his older France stays |
| – | 2026-09-27 08:07 (not uploaded: upload limit reached for the rest of the challenge) | the teammate's newest file (`Friend output/matching_results.tsv`, md5 0f701cbd..., his hold-out 0.99066) merged at x8 (`work/merge_new_x8/`, 97,840,752 bytes, md5 cf52b6e2...; France 868,793 matches; 55,668 of his pairs outside our candidates) | never scored on the leaderboard; `src/run_pipeline.py --mate` reproduces it byte for byte, also from the packaged zip | – |

France is settled: the teammate's French matches as they are (#9, 0.986375) beat both our France with his vote (0.986333) and his France plus our q >= 0.97 French matches (0.986333). Our model's confident French matches are no better than a coin flip where he disagrees. (`output/` now holds #13; see Final package.)

Reading of #9: it differs from the teammate's file only in India/US (our decisions, his vote breaking ties), so our India/US decisions are worth ~+0.001 over his. The pipeline step is now `src/team_merge.py` (writes both TSVs; reproduces #9 exactly, md5 cf76c6f4...). Not yet done: his France vs ours (`work/sub_team_merge/` = our France with his votes, ready for an A/B upload); a trained combination needs his out-of-fold predictions on our validation S1 entities.

Upload budget (26 Sept ~16:45 IST): 3 uploads were left on 26 Sept UTC (until 05:30 IST on 27 Sept), 2 for this pipeline and 1 for a teammate. After #7: 1 left for this pipeline.

## Final package (27 Sept, first version: #13; superseded by #15, see the end of this file)

**The zip packages upload #13, the ranked file** (public LB 0.986435, rank 194). The portal ranks the team by its best upload.
- `output/` holds #13 (98,076,606 bytes, md5 508d340f...; candidates md5 25528912...), identical to its backup in `work/output_backups/lb098644_iu_lr8/`.
- #13 is `src/team_merge.py --mate-unseen --vote-lr 8`, applied to this pipeline's final probabilities and the teammate pipeline's file (`matching_results_ce.tsv`).
- Two pieces are not in the zip:
  - the teammate pipeline's code, which is too large;
  - that exact file, which is no longer on this Mac and was never copied into `work/`. Only our merged outputs were backed up.
  So `output/` cannot be regenerated from the zip alone. The README and the document say so.
- This pipeline alone is upload #8 (0.98515). The zip's code regenerates it byte for byte: stage 2 rerun from an unzipped test build matched md5 38402508..., and the validator printed PASS.
- `work/make_submission_zip.py --team "The Mavericks"` refuses to build if placeholders remain in the docs or if `output/` differs from the uploaded #13 backup.
- The teammate's newest file is backed up in `work/inputs/` (md5 0f701cbd...).
- Final out-of-fold error analysis (v6 + CE2):
  - precision 0.9983, recall 0.9644;
  - oracle on candidates 0.9924;
  - gains available: +0.0078 if blocking lost nothing, +0.0036 if every candidate match were found, +0.0015 with no false matches.

## Two extra uploads (27 Sept, ~17:30 IST)

The organizers gave 2 more uploads and reopened the final submission. Both candidates can be rebuilt exactly from files we have. `team_merge.py` gained `--mate-unseen-file` for this.

| # | File | What it tests | Public LB |
|---|---|---|---|
| 15 | `work/merge_newIU_oldFR_x8/` (98,121,077 bytes, md5 d23b9441...; backup `work/output_backups/lb098682_newIU_oldFR_x8/`) | India/US: the teammate's newest file (hold-out 0.99066) as a x8 vote. France: his old matches as they are (`work/inputs/teammate_v1_france_from_upload13.tsv`, taken from #13, where they were copied unchanged; md5 2df0cab6...) | **0.98682** (best; +0.000385 over #13, from his newest India/US votes) |
| – | `work/merge_newIU_voteFR_x8/` (97,947,298 bytes, md5 e3f7d192...) | #15 with France settled by `src/france_vote.py`, a 2-of-3 vote. It starts from his old France, drops 15,239 pairs his newest model dropped where our q < 0.1, and adds 1,708 pairs his newest added where our q >= 0.725. 16,287 French S1 (6.3%) change; French empty rate goes 5.39% -> 5.64% (train singleton rate 5.58%). India/US are identical to #15 | **withdrawn, not uploaded** (see below) |
| 16 | `work/merge_newIU_v4FR_x8/` (97,985,763 bytes, md5 77179cb7...; candidates md5 ae926b85...) | #15 with the teammate's LATEST France as is: `Friend output/matching_results2.tsv` (md5 a806a079..., backed up as `work/inputs/teammate_matching_results2.tsv`), sent 27 Sept ~19:15. Its India/US rows are identical to the previous file, so #16 differs from #15 only in France: 880,078 French pairs (3.392 per S1; empty 5.56%), against version 1's 890,613 (-24,285 / +13,750); 28,216 French S1 differ from his previous file. `team_merge.py --mate <it> --mate-unseen --vote-lr 8`; validator PASS with `--check-ids`. Backup `work/output_backups/lb0988708_latestFR_x8/` | **0.988708** (best; +0.0019 over #15, from France alone) |

x16 (`work/merge_newIU_oldFR_x16/`) was dropped. A quadratic fit in log(vote) through x1.5 / x3 / x8 peaks near x6, so x16 would more likely lose ~0.0002.

Checks on #15:
- Its France rows are identical to #13's.
- Its India/US rows are identical to `work/merge_new_x8/`.
- Changes vs #13, India: +4,577 / -2,073 pairs (0.80% of S1).
- Changes vs #13, US: +2,831 / -1,889 pairs (0.70% of S1).
- The validator printed PASS.

**Large cross-encoder (Kaggle, xlm-roberta-large, MIT, 560M), scores downloaded 27 Sept 17:40**
- The score files are complete: oof 164,745, test 871,262 and oof_loco 197,275 pairs, all matching the exported pairs.
- **As extra stage-2 features next to CE2** (`--ce-extra-prefix cel`):
  - hard-pair AUC 0.9603 -> 0.9617;
  - CV India 0.9870 -> 0.9871, US 0.9876 -> 0.9877;
  - test mix 0.9873, unchanged at 4 decimals.
- **Replacing CE2** (`--ce-prefix cel`): CE AUC 0.9265 against CE2's 0.9300; CV 0.9872, worse.
- **In the #15 merge**, the extra-feature variant (`work/merge_cel_newIU_oldFR_x8/`) changes only 1,684 India/US entities (0.1%) and no French ones. Not worth an upload.
- The baseline `stage2_oof.parquet` and `stage2.joblib` were restored from `work/stage2_base/`.

**#16 withdrawn (27 Sept ~18:10 IST).** The France vote leans on our low French probabilities, and those are not reliable:
- Our q < 0.02 for 1.6% of the teammate's newest French matches (14,021 pairs), against 0.04% (India) and 0.07% (US) of his India/US matches.
- Even among the 863,599 French pairs that both his versions match, our q < 0.02 for 12,356. His France beats ours on the leaderboard, so these are more likely our errors.
- #14 (a later teammate France with ~17k fewer pairs) scored lower than #13.

`output/` is back to #15 (md5 d23b9441...). The last upload stays unused unless a labelled check shows a gain. That check would be a merge trained on the teammate's held-out predictions, if he can send them.

**Final package = #15** (public LB 0.98682).
- `work/make_submission_zip.py` now expects the #15 backup.
- It adds the two pipeline-B files from `work/inputs/`, as `inputs/pipeline_b_latest_matching_results.tsv` and `inputs/pipeline_b_v1_france_matches.tsv`, with md5 checks.
- `run_pipeline.py --mate inputs/pipeline_b_latest_matching_results.tsv --mate-unseen-file inputs/pipeline_b_v1_france_matches.tsv` regenerates `output/` byte for byte.
- 55,242 of #15's candidates come from pipeline B:
  - 42,615 India/US pairs from the newest file;
  - 12,627 French pairs from version 1.
- In total #15 has 11,483,916 candidates (6.63 per S1) and 5,872,566 matches.

**#16 = 0.988708, the final package (27 Sept ~19:50 IST).**
- #16 differs from #15 only in France: 34,620 French S1 (2.0% of test S1).
  - The gain of +0.0019 means about +0.09 F0.5 per changed French entity.
  - The teammate's latest French matches leave 5.56% of French S1 empty, against 5.39% for his version 1. The training countries' no-match rate is 5.58%.
- At 19:39, `output/` was accidentally overwritten with #15. This was caught before the upload: #16 was copied back and verified (md5 77179cb7...) before it was uploaded.
- The zip now packages #16. `run_pipeline.py --mate inputs/pipeline_b_latest_matching_results.tsv` regenerates it.
  - It holds a single pipeline-B file: `work/inputs/teammate_matching_results2.tsv`, md5 a806a079...
  - The #15 package is kept as `work/submission/The_Mavericks_submission_lb098682_upload15.zip`. The #15 versions of the docs are in `work/staged_upload16/prev_upload15/`.

