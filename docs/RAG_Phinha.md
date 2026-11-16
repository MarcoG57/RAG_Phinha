# B4T: Possession from Incomplete Football Tracking

Marco Giamberduca, Adrián Castaño and Ariel Alvano · RAG_Phinha

For each video frame, B4T estimates which team controls the ball or whether
play is stopped. A frame is one image in the video. The input is automatic
tracking of players and the ball, plus match metadata. The model does not read
the video. This guide explains the approach; the [report](report/RAG_Phinha.pdf)
contains the figures and comparisons.

## Method

B4T combines four model components that learn from movement over time.
Each component has three training runs with different random starting points
(seeds). It estimates team control and the probability of dead ball separately.
The model predicts again with the teams exchanged, maps the result back to the
original teams and averages both views.

A Viterbi decoder chooses the sequence for each half by balancing frame
evidence against the cost of changing labels. H means home control, A away
control and D dead ball. X marks frames outside the scored periods.
Changing team costs 50; entering or leaving D also costs 50.

The selected recipe applies a two-second rule to short control runs in DFL and
older SkillCorner training labels. It does not impose a hard minimum on predicted possessions.
The pipeline applies no fixed D interval or Smart Tagging.
[METHOD](METHOD.md) gives the exact label transformation, equations and parameters.

## Training data

The labelled sources are seven Sportec/DFL matches, 20 current SkillCorner
matches and nine older SkillCorner matches used by one control component.
[SOURCES](SOURCES.md) records their versions, licences, label construction and
development history. [TRAINING](TRAINING.md) contains the training commands.

## Results

DFL evaluation holds out each match in turn: models are trained on the other
matches before predicting it. SkillCorner evaluation uses four groups with
separate training and evaluation matches. These development matches also
informed model choices.

spellF1 scores how well predicted H/A possession intervals match the reference
intervals for the same team. It averages F1 over five time-overlap thresholds:
0.3, 0.4, 0.5, 0.6 and 0.7. The score ranges from 0 to 1; higher is better.
It is not the percentage of frames labelled correctly. D affects interval
boundaries but is excluded from the scored possession objects.

The report compares three input conditions:

| Input | Meaning |
|---|---|
| Clean | External tracking as received. |
| Degraded | `combined_dead`: simulated tracking errors, including ball removal during D intervals identified by external reference labels. |
| 40% gaps | Only long gaps in ball observations, targeting 40% of each period. |

Under Degraded input, the selected recipe reaches spellF1 0.7309 against the
two-second DFL reference and 0.7376 against the three-second reference.
The SkillCorner score is 0.6829. These are development results on external
matches. The score on the challenge match remains unknown.

## Reproduction

Follow [README](../README.md#regenerate-the-submission) to regenerate the CSV
on CPU with the included 24 weight files. Those instructions include the expected
file fingerprint and the validation commands. Two fresh Windows CPU predictions
produced identical files and passed the official validator.

The [LaTeX source](report/main.tex) and
[build instructions](report/README.md) reproduce the report. It records the
original baseline-guided protocol choice and the later post-hoc 50/50 decoder choice.
