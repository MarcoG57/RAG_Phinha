# B4T implementation reference

Read the [technical report](RAG_Phinha.md) for the approach and main evidence.
This reference records implementation settings, development comparisons and
historical provenance. All scores below come from external development matches.
The report and [SOURCES](SOURCES.md#submission-compliance) retain both unresolved
questions about historical selection.

## Network and decoding settings

### Frame probabilities

- *Dead ball:* P(D).
- *Home control:* P(H | alive).
- **Four models**, each with both heads and three seeds (`20261029`, `1`, `2`), so twelve members in all:
  - Model 1: the base recipe.
  - Model 2: the same recipe with extra training copies of the DFL matches, degraded in other ways.
  - Model 3: model 2 plus nine older SkillCorner broadcast matches, which supervise only the team head.
  - Model 4: model 2 plus a fourth degraded copy and three fixes from our code audit.
- Every member is a bidirectional GRU that reads per-frame features of the tracking at 5 Hz, in 60-second windows inside each half.
  - The features describe team shape, direction of attack, advance and forward speed, pressure, player movement and observation coverage.
  - The team head also uses ball features.
- **Test-time team swap:** every member also predicts the match with the teams exchanged and the pitch rotated, and the two views are averaged in log-odds.
- The members' log-odds are averaged with equal weights, and the probabilities are interpolated back to the input frames.

### Sequence decoding

- It runs a single H/A/D decoding per half.
- Per frame, with team evidence *c* and dead evidence *d*, it scores H as *c − d*, A as *−c − d* and D as *d + |c|*. The live/dead decision is therefore the same as in a separate dead-ball decoder; only the switch costs interact.
- Switching team costs 50, and entering or leaving D costs 50. This removes implausibly short flips. D→H and D→A cost the same, so possession is never carried across a stoppage.
- Dead evidence (`mixdead`) combines the learned P(D) with hand-made E2 evidence from the fixed configuration in `configs/model.toml` (the committed card retains the historical source name `configs/e1.toml`): ball out of the pitch, collective stillness and a fast ball. A missing ball counts only together with stillness.
- The runner passes `known_dead=None` and applies no known-dead override.
- During the recurrence, ties keep the previous state. At the final frame,
  tied states are preferred in the order H, A, D.

## Possession-reference transformation

- **Reference used for training.** For DFL and SkillCorner 2019/20, inside an alive segment, a run of H or A shorter than 2 s that follows the other team is given to that other team. The 3-second reference is also used for evaluation. SkillCorner phases of play are treated as already controlled and are not relabelled by this filter.
- **Loose balls and challenges.**
  - There is no separate label for a contested ball. The reference transformation above does not guarantee that every short contest retains the previous team.
  - The decoder needs sustained evidence before it switches team, so a brief contest can stay with the previous team.
  - While the ball is missing, the ensemble keeps the team that shape and pressure support; it does not abstain.
- **Restarts and stoppages.** A restart is the transition from D to live H/A. The model does not classify restart types and applies no special restart rule.
- **X** is written only outside the scored periods.

These are algorithmic conventions, not proof of actual control.

## Training-only transformations

- Every external match is degraded with the `combined_dead` scenario before training: long ball gaps, player dropout, coordinate noise, and the ball removed with probability 0.8 for each stoppage interval.
  - Models 2–4 add training-only copies of the DFL matches: `combined_dead` with another seed and 40 % long ball gaps.
  - Model 4 also adds `dead_ball_missing`.
  - Parameters are fixed in the committed recipes; the historical choice of scenario and reference is qualified in [submission compliance](SOURCES.md#submission-compliance).
- **Team swap.** Team-swap augmentation, which flips the pitch and the H/A labels, is used in training as well as at test time.

## Historical development comparisons (three-second training)

The evaluation is leave-one-match-out over the 7 DFL matches. SkillCorner is used for training and as a second reference with 4 grouped folds. The official scorer is used unmodified. The decoder uses switch costs 50/50; spellF1 is the evaluation metric. Every match is predicted by models that never saw it.

| Model | 3 s | 2 s | Clean input | 40 % ball gaps | SkillCorner |
|---|---:|---:|---:|---:|---:|
| Hand rules E1+E2 | 0.521 | — | — | — | — |
| Base recipe, two-stage decoder (previous candidate A) | 0.719 | 0.700 | 0.698 | 0.663 | 0.666 |
| Four-model ensemble, joint decoder (B4) | 0.736 | 0.723 | 0.751 | 0.725 | 0.670 |
| **Historical B4 with test-time team swap** | **0.741\*** | **0.731\*** | **0.753** | **0.727** | **0.679** |

The historical 2 s column uses recipes separately retrained with the two-second label transformation. It is not the three-second weights merely scored against a different reference. Clean-input and 40% gap columns use the historical three-second reference. The next section reports the selected two-second recipe against multiple references explicitly.

\* **Selection bias.** These figures are optimistic, for three reasons:
- the costs and recipes were chosen on these same 7 matches;
- the ensemble is the best of four pre-registered combinations;
- several further pre-registered changes were tried and only one, the test-time swap, was adopted.

There is no untouched external test set. BAR-ATH is a different provider and camera, and we expect a lower score on the real match.

- **Against the base recipe,** B4T wins on 7 of 7 DFL matches at 3 s (+0.022) and at 2 s (+0.031), on clean input and with 40 % ball gaps. It also wins on 15 of 20 SkillCorner matches (+0.013).
- **How the test-time swap was adopted:** it was pre-registered with a two-stage rule, and adopted after it improved the ensemble on all five references.
- **On scenarios not used for any choice,** the ensemble without test-time swap also wins on 7 of 7 matches:
  - degradation without the label-driven removal of the ball at stoppages: 0.734 vs 0.711;
  - players swapped between teams: 0.744 vs 0.686.

  These robustness figures were measured without the test-time swap.
- **Decoder costs:** across a 20-pair grid of decoder costs, the ensemble without test-time swap is better on DFL at every pair, and at parity on SkillCorner.

## Two-second candidate development results

The selected two-second-trained candidate has degraded DFL spellF1 0.560711 against raw, 0.730868 against 2 s and 0.737603 against 3 s references. Clean DFL scores are 0.7426 (2 s) and 0.7346 (3 s); 40% gaps scores are 0.7145 (2 s) and 0.7121 (3 s). SkillCorner is 0.682904. These are selection-biased external development results, not private scores.

E15 Amendment 2 retained joint decoder costs 50/50 after observing the results.
The registered rule had selected 12.5/12.5. This post-hoc choice trades some
fine-reference mean score for smaller losses against the three-second reference.
The cost choice followed the results; the score difference remains measured,
without a statistical test that attributes it to noise.

## Experiments not selected

Everything here was measured on external data only.

| Idea | Result |
|---|---|
| Visibility gating and hand/learned hybrids | Below our +0.010 threshold |
| Hand-made dead evidence alone | −0.025 |
| Single models trained with extra degraded copies | Robust, but −0.008 to −0.018 on SkillCorner (kept only inside the ensemble) |
| A second-stage gradient-boosting layer over the model outputs | Worse on clean input and with 40 % ball gaps |
| Relabelling suspected stoppages inside SkillCorner phases | ≈ 0: changes only 0.33 % of the H/A frames |
| A 5-seed single model; a 12.5 Hz GRU | ≈ 0; better at one reference, worse at the other |
| Pseudo-labels, simulated on external matches only | −0.002 |
| Triangular window weights; nested decoder-cost selection; E2 weight (E9–E11) | −0.003, −0.004 and −0.001 |
| Variable H/A switch cost from a learned "possession changes here" head, `50·max(1 − p, 0.1)` | 0.7107 vs 0.7164 with the constant cost; won 2 of 7 matches (criterion: +0.010 and 5 of 7) |
| A degradation scenario with constants from aggregate counts of the provided tracking | Withdrawn under the 3 November clarification |

## Artifact trust and historical reproduction

- `scripts/check_repo.py` checks the pinned SHA-256 of the card and all 24 weight files. Inference authenticates the selected card against its independent pin, validates versions and checks the exact weight bytes before deserializing them. Locally retrained models require an explicitly trusted card hash; hashes do not make an unknown checkpoint safe.
- `--check-repeat` predicts twice with fresh model loads and requires identical bytes.
- The runner checks the official checkout before loading its reader or predicting, and validates the final CSV again before publication to its local path.
- `--expected-sha256` refuses to finalize the CSV unless it matches `3eea6c25…`, reproduced on three Windows machines on 16 November with identical repeat bytes and official validation. Two runs used the delivered prediction path, including one from a fresh installation. The third used a development branch with Smart Tagging and produced the same bytes as the unmasked candidate.
- Casti reported another matching repeat and official validation from a fresh Debian Docker environment: Python 3.11.18, PyTorch 2.5.1+cpu and one thread. It used the earlier source with the same prediction functions and weights, before the new early checkout check. This establishes a reported reproduction in that environment, not universal Linux compatibility.
- **Retraining.** The training commands for this delivery are in [TRAINING](TRAINING.md). The selected card records the component training commits and commands. Prediction needs only the submission commit. A delivery snapshot with a single commit does not include that training history; the recorded commits establish provenance, not availability of the original training checkouts in that snapshot. Training on a GPU is not guaranteed to be bit-exact.

These runs cover the two-second candidate's predictions in the stated
environments. The current documentation was reviewed locally. Commands and
recovery steps are in [README](../README.md); training and evaluation commands
are in [TRAINING](TRAINING.md). [SOURCES](SOURCES.md#submission-compliance)
records source versions and the development history.
