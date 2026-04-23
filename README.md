# Liver Meta Calculator

Research-stage machine learning project and local web/API application for estimating the risk of advanced liver fibrosis.

The project predicts the probability of advanced fibrosis, defined as:

- positive class: `FSCORE 3-4`
- negative class: `FSCORE 1-2`

The model returns a calibrated probability `P(FSCORE 3-4)` and maps it into a triage-oriented interpretation: `OUT`, `IN`, or `GREY`.

> This repository is not a certified medical device and must not be used as a substitute for clinical judgment.

## Project Overview

`liver-meta-calculator` is an end-to-end ML and application project developed from thesis-stage research on non-invasive liver fibrosis risk stratification.

The repository contains two clearly separated parts:

1. **Research / ML layer**
   - feature engineering based on established hepatology calculators,
   - model training and calibration,
   - threshold selection for triage logic,
   - reports, plots, and archived experiments.

2. **Application layer**
   - FastAPI inference API,
   - Pydantic request/response schemas,
   - local Jinja2-based web interface,
   - Dockerized runtime,
   - automated tests and linting setup.

The current application is designed for local experimentation, portfolio review, and further ML engineering work. It is not intended for direct clinical deployment.

## Key Features

- Binary classification target: `FSCORE 3-4` vs `FSCORE 1-2`.
- Uses classic hepatology calculators as meta-features:
  - `APRI`
  - `FIB-4`
  - `NAFLD Fibrosis Score`
  - `de Ritis / RITIS`
- Uses additional clinical/laboratory inputs, including:
  - age,
  - platelet count,
  - AST,
  - ALT,
  - albumin.
- Returns calibrated probability `P(FSCORE 3-4)`.
- Converts probability into triage zones:
  - `OUT` / rule-out,
  - `IN` / rule-in,
  - `GREY` / uncertain zone.
- Provides a FastAPI API and local browser UI.
- Includes Docker / Docker Compose support.
- Includes tests and linting configuration.

## Clinical/Research Idea

Standalone fibrosis calculators are useful, but each has limitations. This project treats them not as final decision rules, but as informative features for a meta-model.

The central research idea is:

1. calculate or provide established non-invasive fibrosis scores,
2. combine them with selected laboratory and demographic variables,
3. estimate calibrated probability of advanced fibrosis,
4. apply two thresholds to support triage instead of forcing every patient into a binary decision.

This design explicitly separates low-risk, high-risk, and uncertain cases.

Thesis-stage experiments suggested that a calibrated random forest could achieve ROC AUC around `0.88`. Outside the grey zone, reported experimental metrics were approximately:

- sensitivity: `0.947`
- specificity: `0.740`
- F1: `0.853`

These values are research-stage results only. They should not be interpreted as clinical validation.

## Model Inputs and Output

The `/predict` endpoint expects the following input fields:

| Field | Description |
| --- | --- |
| `age` | Patient age |
| `platelets_k_per_ul` | Platelet count in thousands per microliter |
| `ast_u_l` | AST level in U/L |
| `alt_u_l` | ALT level in U/L |
| `albumin_g_l` | Albumin level in g/L |
| `fib4` | FIB-4 score |
| `apri` | APRI score |
| `ritis` | de Ritis / RITIS ratio |
| `nafld` | NAFLD Fibrosis Score |

The model output contains:

| Field | Description |
| --- | --- |
| `model_name` | Name of the loaded model metadata |
| `positive_label` | Positive class label, expected `3-4` |
| `negative_label` | Negative class label, expected `1-2` |
| `probability_positive` | Calibrated probability of `FSCORE 3-4` |
| `triage_zone` | One of `OUT`, `IN`, `GREY` |
| `threshold_out` | Rule-out threshold loaded from model metadata |
| `threshold_in` | Rule-in threshold loaded from model metadata |

## Triage Logic

The application uses two probability thresholds:

- `t_out`: rule-out threshold
- `t_in`: rule-in threshold

The interpretation is:

```text
if P(FSCORE 3-4) < t_out:
    triage_zone = OUT

if P(FSCORE 3-4) >= t_in:
    triage_zone = IN

otherwise:
    triage_zone = GREY
```

The grey zone is intentional. It represents cases where the model should not force a confident low-risk or high-risk interpretation.

## Repository Structure

```text
.
├── src/liver_calculator/              # Main application package
│   ├── api/                           # FastAPI application and routes
│   ├── web/                           # Local web UI, templates, static assets
│   ├── services/scoring.py            # Model loading, feature mapping, prediction logic
│   ├── config.py                      # Environment-driven app/model configuration
│   └── schemas.py                     # Pydantic request/response schemas
├── scripts/
│   ├── serve_api.py                   # Run local API/web UI
│   └── train_meta_logistic.py         # Train current logistic model candidate
├── models/                            # Model artifacts and metadata
├── reports/                           # Metrics, figures, and reports
├── data/                              # Local private data, not for publication
├── tests/                             # Automated tests
├── docs/                              # Project documentation
├── references/                        # Supporting materials
├── legacy/                            # Archived research experiments
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
└── README.md
```

## Installation

Requirements:

- Python `>=3.10`
- pip
- optional: Docker and Docker Compose

Create and activate a virtual environment:

```bash
python -m venv .venv
. .venv/Scripts/activate
```

On PowerShell, use:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the project with development and training dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -e .[dev,train]
```

## Running Locally

Start the local API and web UI:

```bash
python scripts/serve_api.py
```

Then open:

```text
http://127.0.0.1:8000
```

Available local routes:

- `GET /` - local web interface
- `GET /health` - application and model readiness status
- `GET /model-info` - loaded model metadata summary
- `POST /predict` - prediction endpoint
- `GET /docs` - interactive OpenAPI documentation

The default runtime configuration can be adjusted with environment variables:

```text
APP_TITLE=Liver Meta Calculator
APP_HOST=0.0.0.0
APP_PORT=8000
APP_RELOAD=false
MODEL_NAME=meta_logistic
MODEL_PATH=models/lr_calibrated_triage_model.joblib
MODEL_METADATA_PATH=models/metadata/lr_calibrated_triage_meta.json
```

## Docker Usage

Build and run the application with Docker Compose:

```bash
docker compose up --build
```

Then open:

```text
http://127.0.0.1:8000
```

The container exposes the FastAPI app on port `8000` by default. Model artifacts are mounted from the local `models/` directory.

## API Usage

Example request:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 55,
    "platelets_k_per_ul": 180,
    "ast_u_l": 42,
    "alt_u_l": 38,
    "albumin_g_l": 43,
    "fib4": 2.1,
    "apri": 0.7,
    "ritis": 1.1,
    "nafld": -0.4
  }'
```

Example response shape:

```json
{
  "model_name": "meta_logistic",
  "positive_label": "3-4",
  "negative_label": "1-2",
  "probability_positive": 0.31,
  "triage_zone": "GREY",
  "threshold_out": 0.16,
  "threshold_in": 0.34
}
```

The numeric values above are illustrative. Actual probabilities and thresholds depend on the loaded model artifact and metadata.

## Training

The current training entrypoint is:

```bash
python scripts/train_meta_logistic.py
```

The script trains the current calibrated logistic model candidate, writes model artifacts to `models/`, and writes reports/plots to `reports/`.

Important notes:

- Training requires local private medical datasets.
- Those datasets are not included in the public repository.
- A clean public clone may be able to run inference if model artifacts are present, but it should not be expected to reproduce training without access to the private data files.
- Paths and model metadata should be reviewed before running new experiments.

## Testing and Linting

Run automated tests:

```bash
pytest
```

Run linting:

```bash
ruff check src tests scripts
```

These checks cover the API, scoring logic, and web interface behavior.

## Data Availability

Patient-level medical data used for model development is private and cannot be published in this repository.

For that reason:

- raw clinical data is not public,
- processed patient datasets are not public,
- training data is not distributed with the repository,
- reproducibility from raw data is limited for external users.

The repository is structured so that application code, model-serving logic, metadata, tests, and documentation can be reviewed without exposing private patient records.

## Limitations

- This is a research-stage project, not a clinically validated product.
- The model is not a certified medical device.
- Reported metrics come from thesis-stage or local experimental evaluation.
- External validation is limited.
- Data availability is restricted because the source dataset contains private medical information.
- Model behavior depends on the representativeness and quality of the private training data.
- The grey zone means some cases deliberately remain unresolved by the model.
- Code in `legacy/` may reflect older exploratory experiments and should not be treated as the current production path.

## Medical Disclaimer

This project is provided for research, educational, and software engineering portfolio purposes only.

It must not be used to diagnose, treat, prevent, or rule out disease in real patients. It is not a replacement for physician judgment, specialist hepatology assessment, laboratory interpretation, imaging, elastography, biopsy, or any other clinically approved diagnostic process.

The model is not certified, not externally validated for clinical deployment, and not approved as a medical device.

## Roadmap

Planned or possible next steps:

- improve public documentation of model metadata and inference assumptions,
- add stronger validation around input ranges and units,
- add versioned model cards for released artifacts,
- separate research notebooks and archived experiments more clearly from the serving path,
- add CI checks for tests, linting, and Docker build,
- improve calibration and threshold reporting,
- add external validation if an appropriate dataset becomes available,
- prepare a safer demo mode using synthetic examples only,
- document deployment constraints and privacy requirements.
