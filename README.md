# Liver Meta Calculator

A structured machine learning project for liver fibrosis risk scoring. The repository combines a data-science-friendly layout with an API-ready application layer so it can grow from research code into a small web app.

## Current status

- The repository has been reorganized into a cleaner GitHub structure.
- The current deployment candidate is the logistic meta-calculator training script at `scripts/train_meta_logistic.py`.
- Historical experiments, reports, and exploratory scripts were preserved under `legacy/` instead of being kept in the repository root.

## Repository layout

```text
.
├── .github/workflows/     # CI for linting and tests
├── data/                  # Datasets kept from the original project
├── docs/                  # Project documentation
├── legacy/                # Archived research code and old results
├── models/                # Serialized models and model metadata
├── notebooks/             # Future notebooks for EDA and experiments
├── references/            # Clinical notes, business rules, external references
├── reports/               # Metrics, figures, and analysis outputs
├── scripts/               # Executable entrypoints for training and local serving
├── src/liver_calculator/  # Installable Python package for inference and API
└── tests/                 # Automated tests
```

## Quick start

```bash
python -m venv .venv
. .venv/Scripts/activate
python -m pip install --upgrade pip
python -m pip install -e .[dev,train]
```

Run the API locally:

```bash
python scripts/serve_api.py
```

Retrain the current logistic meta-calculator:

```bash
python scripts/train_meta_logistic.py
```

## Project direction

The target product is a simple web application where a user enters clinical values and receives a score in the fibrosis scale. The repository is now prepared for three parallel tracks:

- data work and reproducible experiments,
- model training and artifact management,
- API/web application development on top of the selected model.

## Notes

- `legacy/` keeps the original research material with minimal interference.
- The `data/` directory still contains the historic dataset hierarchy; it can be refactored further once a stable ingestion pipeline is defined.
- A license file was intentionally not added because that should follow your publishing decision.
