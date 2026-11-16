# RAG_Phinha delivery code

Read README.md, docs/RAG_Phinha.md and docs/SOURCES.md before changing this package.
This repository contains training, evaluation and inference for B4T only.

Use README.md for the reader's setup, inference and recovery path. The technical
report in docs/RAG_Phinha.md is the short review path; docs/METHOD.md keeps exact
implementation details and historical comparisons. docs/TRAINING.md owns the
training/evaluation commands; docs/SOURCES.md owns provenance and compliance.
Keep one maintained source for each command or constraint and link to it.

- Preserve the selected card, its 24 weight files and independent SHA256 pins.
- Preserve gru11/gru11b features, component order, joint decoding, TTA and CSV bytes.
- Train, tune and choose parameters only on licensed external data. Keep all
  copies of a held-out match outside every component's training partition.
- ORG-2026-11-03 takes precedence over earlier guidance. Do not compare BAR-ATH
  outputs with video, create manual match data, or upload supplied assets to AI.
- Keep data and outputs ignored. Do not overwrite runs or suppress failed checks.
- Run `python scripts/check_repo.py --tracked` and `python -m unittest discover -s tests -v`.
  Changes to inference require exact CPU CSV reproduction with repeat and expected hash.
- The historical protocol choice and public distribution of weights remain
  open questions; the model does not apply a D mask. A software PASS does not
  clear them. See docs/SOURCES.md for the selection history and compliance limits.
