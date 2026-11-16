# RAG_Phinha: B4T possession estimation

B4T estimates who controls the ball in each of the 199,319 BAR-ATH video frames.
A frame is one image in the video. The labels are H for the home team,
A for the away team, D for dead ball, and X outside the scored periods.
The input is automatic tracking of players and the ball, plus match metadata.
The pipeline uses no video, fixed D interval or Smart Tagging.

You can reproduce the CSV on a standard computer processor (CPU), without a
graphics card or new training. The 24 included weight files contain the learned
model parameters. The model card records their configuration and provenance.
Training and evaluation code are also included. Match data, generated CSVs and
training outputs stay outside the repository.

| I want to | Start here |
|---|---|
| Understand the method | [Reading guide](docs/RAG_Phinha.md) · [PDF](docs/report/RAG_Phinha.pdf) · [LaTeX source](docs/report/main.tex) |
| Reproduce the CSV on CPU | [Install](#install), then [run inference](#regenerate-the-submission) |
| Inspect how the method is implemented | [Implementation guide](docs/METHOD.md) |
| Train or evaluate the model | [Training and evaluation](docs/TRAINING.md) |
| Check sources, rights and rules | [Sources and compliance](docs/SOURCES.md) |

## Install

Use Git and Python 3.11. Run from a Git clone: the scripts read its commit and
tracked files. The source ZIP contains the files for inspection, without Git metadata.

On Windows, create a virtual environment and install the pinned CPU dependencies:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.5.1+cpu --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.lock.txt
.\.venv\Scripts\python.exe -m pip check
```

On Linux, the equivalent environment setup is:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install torch==2.5.1+cpu --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -r requirements.lock.txt
.venv/bin/python -m pip check
```

On macOS, install `requirements.lock.txt` without the separate `+cpu` wheel command.
Exact reproduction on macOS is untested. The lock retains the versions recorded
by the model card, including the training libraries.

Prepare the unmodified official reader and validator once:

```powershell
git -c core.autocrlf=false clone https://github.com/metrica-sports/more-than-a-hack-2027 .official
git -C .official checkout f8f2629ff1e0985f81d7e59d8413678514ee1530
```

If `.official` already exists, check its commit and tracked files before continuing.

On Linux, prepare `.official` with LF line endings. A checkout copied or mounted
from Windows can fail with `official tracked files modified` because of CRLF.
Keep that checkout and prepare a fresh LF clone from Linux with the command above.

## Regenerate the submission

Obtain the tactical dataset through the authorised channel. Its root must contain
`tactical/BAR-ATH_tactical_atd.xml` and `tactical/BAR-ATH_tactical_atd.txt`.

```powershell
$dataset = 'C:\path\to\dataset'
$run = 'submission-01' # Choose a new value for each attempt, including retries.
.\.venv\Scripts\python.exe -B scripts/check_repo.py --tracked
if ($LASTEXITCODE -ne 0) { throw 'Repository integrity failed.' }
.\.venv\Scripts\python.exe -B scripts/predict_final.py --model models/casti-ens4-joint-tta-c2 --data-dir $dataset --name $run --check-repeat --expected-sha256 3eea6c2580fc4ef4e36e7d70987534ddaba0b7ca86cd40e11f4775c3f6d33073
if ($LASTEXITCODE -ne 0) { throw 'Inference failed; inspect its manifest before retrying.' }
.\.venv\Scripts\python.exe -B scripts/check_submission.py "artifacts/final/$run/submission.csv"
if ($LASTEXITCODE -ne 0) { throw 'CSV validation failed.' }
```

Choose a new `--name` for every attempt, including retries after a failure.
SHA-256 is a file fingerprint: identical hashes confirm the expected CSV bytes.
Accept only exit 0, a PASS run record (`manifest.json`), identical repeat bytes
and the expected hash.
The CSV appears at its final local path only after validation. CSVs and manifests
remain under ignored `artifacts/`. [Reproduction results](#reproduction-results)
records the environments tested.

If a command fails, stop and read the error. If the run created
`artifacts/final/<name>/manifest.json`, read its status and error details.
A failure before directory creation can leave no manifest. Check the dataset
paths, official checkout, dependencies or integrity error named in the failure.
Preserve the failed run. Do not use a partial CSV or replace the expected hash
to accept a mismatch.

The equivalent inference command in Bash is:

```bash
dataset='/path/to/dataset'
run='submission-01' # Change for every attempt.
.venv/bin/python -B scripts/check_repo.py --tracked &&
.venv/bin/python -B scripts/predict_final.py --model models/casti-ens4-joint-tta-c2 --data-dir "$dataset" --name "$run" --check-repeat --expected-sha256 3eea6c2580fc4ef4e36e7d70987534ddaba0b7ca86cd40e11f4775c3f6d33073 &&
.venv/bin/python -B scripts/check_submission.py "artifacts/final/$run/submission.csv"
```

Model files use PyTorch serialization and can execute code when loaded. The
selected card is authenticated against its independent repository pin before
any weight is loaded. Load other models through the
[local-model trust procedure](docs/TRAINING.md#trust-a-locally-retrained-model).

## Prepare the local delivery files

After a successful run, copy the CSV into a new local folder with the
required team name. In the same PowerShell session:

```powershell
$delivery = "artifacts/delivery/$run"
if (Test-Path $delivery) { throw 'Choose a new delivery folder.' }
New-Item -ItemType Directory -Path $delivery -ErrorAction Stop | Out-Null
Copy-Item "artifacts/final/$run/submission.csv" "$delivery/RAG_Phinha.csv" -ErrorAction Stop
.\.venv\Scripts\python.exe -B scripts/check_submission.py "$delivery/RAG_Phinha.csv"
if ($LASTEXITCODE -ne 0) { throw 'Final delivery validation failed.' }
if ((Get-FileHash "$delivery/RAG_Phinha.csv" -Algorithm SHA256).Hash.ToLowerInvariant() -ne '3eea6c2580fc4ef4e36e7d70987534ddaba0b7ca86cd40e11f4775c3f6d33073') { throw 'Final CSV hash differs.' }
```

In Bash, create the parent separately and stop on any error:

```bash
mkdir -p artifacts/delivery &&
mkdir "artifacts/delivery/$run" &&
cp "artifacts/final/$run/submission.csv" "artifacts/delivery/$run/RAG_Phinha.csv" &&
.venv/bin/python -B scripts/check_submission.py "artifacts/delivery/$run/RAG_Phinha.csv" &&
printf '%s  %s\n' '3eea6c2580fc4ef4e36e7d70987534ddaba0b7ca86cd40e11f4775c3f6d33073' "artifacts/delivery/$run/RAG_Phinha.csv" | sha256sum --check
```

The form requests `RAG_Phinha.csv`, `RAG_Phinha.pdf` and a public repository URL.
The [report guide](docs/RAG_Phinha.md) links to its
[LaTeX source](docs/report/main.tex) and [build instructions](docs/report/README.md).

The [submission contract](https://github.com/metrica-sports/more-than-a-hack-2027/blob/f8f2629ff1e0985f81d7e59d8413678514ee1530/SUBMISSION.md)
requires one form response per team. Each uploaded file is limited to 10 MB.
Source versions, licences and development history are in
[SOURCES](docs/SOURCES.md#submission-compliance).

## Reproduction results

Three Windows machines reproduced the expected CSV hash on 16 November, with
identical repeat bytes and official validation (Python 3.11.1 and 3.11.9).
Two used the delivered inference path, including one fresh installation following
this README. The third used a development branch with Smart Tagging and produced
the same bytes as the unmasked candidate.

Casti reported the same checks from a fresh Debian Docker installation
(Python 3.11.18, PyTorch 2.5.1+cpu, one thread). That run used the same prediction
functions and weights, before the wrapper's early checkout validation was added.

## Train and evaluate

[Training and evaluation](docs/TRAINING.md) describes the three external sources,
four component commands and leave-one-match-out evaluation. These operations can
be expensive. Inference does not require retraining. GPU retraining is not
expected to reproduce the committed weights bit for bit.

## Verify the software

On Windows:

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -B scripts/check_repo.py --tracked
```

On Linux:

```bash
.venv/bin/python -B -m unittest discover -s tests -v &&
.venv/bin/python -B scripts/check_repo.py --tracked
```

Tests use synthetic data and temporary weights. They cover data alignment,
features, training separation, inference, hashes, failure recovery and export;
they do not establish sports accuracy. `check_repo.py` also checks local links
and the card/weights against independent pins, including the Git index.

## Files and limits

- `src/possession/`: data readers, features, GRU training/prediction, decoding and scoring.
- `scripts/`: acquisition, training, ensemble construction, evaluation and CSV validation.
- `models/casti-ens4-joint-tta-c2/`: the selected model only.
- [Technical report](docs/RAG_Phinha.md) and [LaTeX source](docs/report/main.tex): method, development results and limitations.
- [Sources](docs/SOURCES.md): fixed dataset versions and provenance boundaries.

Original code: [Apache 2.0](LICENSE), with the three authors credited in
[NOTICE](NOTICE). [Third-party notices](THIRD_PARTY_NOTICES.md) remain applicable.
The 24 selected two-second weights have a separate [Apache 2.0 declaration](WEIGHTS_LICENSE.md)
by the three authors, with DFL and SkillCorner attribution in NOTICE. Dataset
and challenge-asset rights remain separate.
