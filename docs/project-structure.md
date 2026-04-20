# Project Structure

This repository now follows a split that is common in public machine learning projects:

- `src/liver_calculator/` contains code that should become reusable and deployable.
- `scripts/` contains command-style entrypoints for training and local serving.
- `data/`, `models/`, and `reports/` hold project artifacts in a way that is easy to understand on GitHub.
- `legacy/` keeps the original experimentation history without polluting the root of the repository.

## Why keep `legacy/`

The original repository contains useful work, but most of it is exploratory and file-path driven. Moving it into `legacy/` keeps the research trace while making the main repository easier to navigate for:

- recruiters and hiring managers,
- collaborators,
- future API or app work,
- future refactors into modular training pipelines.

## Recommended next refactor

The next technical step should be extracting the reusable parts of `scripts/train_meta_logistic.py` into the `src/liver_calculator/` package:

1. dataset loading,
2. preprocessing pipeline creation,
3. threshold selection,
4. training and artifact export.
