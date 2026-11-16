# Train and evaluate B4T

To regenerate the CSV with the included 24 weights, follow the
[CPU inference steps](../README.md#regenerate-the-submission).

This guide covers training and external evaluation. Use the environment and
pinned official checkout from [README](../README.md). Prepare the three licensed
external sources, train four components, combine them, then evaluate held-out
matches. Source identities and partition separation are checked before fitting.
Data and generated models stay ignored. BAR-ATH is reserved for inference.

## Prepare the three sources

Read the licences and versions in [SOURCES](SOURCES.md) before acquisition.
The included downloaders require an explicit download or licence flag; these
commands can fetch large files and are not part of installation or tests.

```powershell
.\.venv\Scripts\python.exe scripts/download_dfl.py --out data/external/dfl --all --accept-license CC-BY4
.\.venv\Scripts\python.exe scripts/fetch_skillcorner.py --download --out data/external/skillcorner
```

The first produces `data/external/dfl/manifest.toml`; the second produces
`data/external/skillcorner/manifest-extrapolated.toml` with 20 fixed matches.
Use the seven DFL matches recorded in the model card. Check acquisition receipts,
hashes and source licences before training.

For the nine SkillCorner 2019/20 matches, obtain the full `match_data.json` and
`structured_data.json` files from commit
`1ed50b775f93075c2a6d00d0d795f13b7b89709e`, with its MIT notice. Place that
checkout under `data/external/skillcorner2019/`, or adjust paths in
[the manifest template](../configs/skillcorner-2019.example.toml). It lists the
nine IDs recorded by the selected component. The format identifier
`skillcorner_legacy` identifies this dataset's schema. Unknown possession is X
and is excluded from D supervision.

## Train the four components

Training can use the GPU build of the pinned PyTorch version. CSV reproduction
uses CPU and the committed weights. Retraining follows the selected recipe,
with possible numerical differences on GPU. The commands write to new model
directories. Choose new names for retries.

| Component | Recipe | Difference | Output directory |
|---|---|---|---|
| Base | `gru11` | `combined_dead` | `models/b4t-base` |
| Copies | `gru11` | Additional `combined_dead:1` and `ball_long_gaps_40:1` copies | `models/b4t-copies` |
| 2019 data | `gru11` | Copies plus the SkillCorner 2019/20 manifest | `models/b4t-data2019` |
| Audit | `gru11b` | Copies plus `dead_ball_missing:1` | `models/b4t-audit` |

On Windows, set the common arguments once:

```powershell
$common = @(
  '--manifest', 'data/external/dfl/manifest.toml',
  '--extra-manifest', 'data/external/skillcorner/manifest-extrapolated.toml',
  '--controlled-s', '2', '--control-cost', '50', '--dead-cost', '50',
  '--seed', '0', '--possession-weights', '--ensemble-seeds', '20261029', '1', '2'
)
.\.venv\Scripts\python.exe scripts/train_final.py @common `
  --recipe gru11 --train-scenario combined_dead --name b4t-base
if ($LASTEXITCODE -ne 0) { throw 'Base component failed.' }
.\.venv\Scripts\python.exe scripts/train_final.py @common `
  --recipe gru11 --train-scenario combined_dead combined_dead:1 ball_long_gaps_40:1 `
  --extra-train-scenario combined_dead --name b4t-copies
if ($LASTEXITCODE -ne 0) { throw 'Copies component failed.' }
.\.venv\Scripts\python.exe scripts/train_final.py @common `
  --extra-manifest configs/skillcorner-2019.example.toml --recipe gru11 `
  --train-scenario combined_dead combined_dead:1 ball_long_gaps_40:1 `
  --extra-train-scenario combined_dead --name b4t-data2019
if ($LASTEXITCODE -ne 0) { throw '2019 component failed.' }
.\.venv\Scripts\python.exe scripts/train_final.py @common `
  --recipe gru11b --train-scenario combined_dead combined_dead:1 ball_long_gaps_40:1 dead_ball_missing:1 `
  --extra-train-scenario combined_dead --name b4t-audit
if ($LASTEXITCODE -ne 0) { throw 'Audit component failed.' }
.\.venv\Scripts\python.exe scripts/combine_models.py . --name b4t-retrained `
  models/b4t-base models/b4t-copies models/b4t-data2019 models/b4t-audit
if ($LASTEXITCODE -ne 0) { throw 'Model combination failed.' }
```

The combination uses equal component weights, joint H/A/D decoding and team-swap
inference. Components must have the same number of members and compatible
metadata; duplicated components and altered weight hashes are rejected.
`gru11b` changes the specified ball-speed feature, overlap/window averaging and
D-loss weighting. Both recipes remain necessary for this selected ensemble.

On Linux, the equivalent Bash commands are below. `set -e` stops this shell
sequence if a training or combination command fails.

```bash
set -e
common=(
  --manifest data/external/dfl/manifest.toml
  --extra-manifest data/external/skillcorner/manifest-extrapolated.toml
  --controlled-s 2 --control-cost 50 --dead-cost 50
  --seed 0 --possession-weights --ensemble-seeds 20261029 1 2
)
.venv/bin/python scripts/train_final.py "${common[@]}" \
  --recipe gru11 --train-scenario combined_dead --name b4t-base
.venv/bin/python scripts/train_final.py "${common[@]}" \
  --recipe gru11 --train-scenario combined_dead combined_dead:1 ball_long_gaps_40:1 \
  --extra-train-scenario combined_dead --name b4t-copies
.venv/bin/python scripts/train_final.py "${common[@]}" \
  --extra-manifest configs/skillcorner-2019.example.toml --recipe gru11 \
  --train-scenario combined_dead combined_dead:1 ball_long_gaps_40:1 \
  --extra-train-scenario combined_dead --name b4t-data2019
.venv/bin/python scripts/train_final.py "${common[@]}" \
  --recipe gru11b --train-scenario combined_dead combined_dead:1 ball_long_gaps_40:1 dead_ball_missing:1 \
  --extra-train-scenario combined_dead --name b4t-audit
.venv/bin/python scripts/combine_models.py . --name b4t-retrained \
  models/b4t-base models/b4t-copies models/b4t-data2019 models/b4t-audit
```

Stop on a nonzero exit code in PowerShell too; inspect the failed run before
choosing new names. Do not combine partial or failed components.

### Trust a locally retrained model

The inference command trusts the committed B4T card through a separate source
pin. A different card requires `--trusted-card-sha256`. After your own successful
training and combination, record its digest while the files are still under your
control. In PowerShell:

```powershell
$trustedCard = (Get-FileHash models/b4t-retrained/card.json -Algorithm SHA256).Hash.ToLowerInvariant()
.\.venv\Scripts\python.exe scripts/predict_final.py --model models/b4t-retrained --trusted-card-sha256 $trustedCard --data-dir $dataset --name retrained-01 --check-repeat
```

In Bash use `sha256sum models/b4t-retrained/card.json` to record the digest, then
pass that recorded value to the same flag. The selected B4T CSV hash applies only
to the committed weights, not to a new training run. Verify source, access rights
and every training step before trusting artifacts obtained from someone else.
Calculating a hash after downloading an unknown card does not authenticate it.
PyTorch checkpoint deserialization is trusted code execution, including on the
pinned 2.5.1 runtime; this procedure establishes trust, not an untrusted-file
sandbox. See the [PyTorch advisory](https://github.com/pytorch/pytorch/security/advisories/GHSA-53q9-r3pm-6pq6).

The original card retains the historical training commands and commits. This
package includes maintained training code for those two recipes, while the
historical commits and research workspace are not part of this delivery snapshot.
The shared hand evidence and decoder configuration are in
[model.toml](../configs/model.toml); the committed card retains its original
metadata source names and embedded evidence constants unchanged.

## Evaluate without match leakage

This command evaluates the historical three-second training recipe. It does not
reproduce the selected two-second candidate's scores. It trains four components
for each DFL held-out match. All copies of that match are excluded from every component's
training before fitted preprocessing. Current SkillCorner is train-only;
SkillCorner 2019/20 is train-only for the third component's H/A head.

Inspect the audited plan before starting the expensive run:

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_model.py --manifest data/external/dfl/manifest.toml --extra-manifest data/external/skillcorner/manifest-extrapolated.toml --legacy-manifest configs/skillcorner-2019.example.toml --name b4t-plan --plan-only
```

On Linux, replace `.\.venv\Scripts\python.exe` with `.venv/bin/python` in the
equivalent command and use the same arguments.

For seven DFL matches and three seeds this means 28 component fits and 168 GRU
heads, each with 4,000 steps. Actual elapsed time and memory are machine-dependent
and have not been measured for a full run of this maintained command.

Run the same command without `--plan-only`, using a new name such as
`b4t-evaluation`. Defaults are the 3-second reference and the `clean`,
`combined_dead` and `ball_long_gaps_40` scenarios. The unmodified official scorer
receives the references, X labels and period boundaries. Results are written to
`artifacts/evaluation/<name>/metrics.json` with per-match aggregates and a
RUNNING/PASS/FAIL manifest. A plan is marked PLAN_ONLY and is not a measured score.

The report gives development scores. Those matches also informed recipe and
cost selection. Use independent matches for a test estimate.
