# Liver Meta Calculator

I built this project as part of my Bachelor's thesis and I developed the entire concept, modeling workflow, code, and repository structure myself. The project is my attempt to go beyond standalone hepatology calculators and turn them into a single calibrated meta-calculator for liver fibrosis risk stratification.

My long-term goal is to turn this into a simple web application where a user enters clinical values and receives a probability-based fibrosis assessment together with a triage-style interpretation.

## What This Project Is

I designed this project as an end-to-end machine learning workflow for distinguishing between:

- `FSCORE 1-2`
- `FSCORE 3-4`

Instead of relying on one fixed clinical score, I combine multiple non-invasive hepatology calculators and selected laboratory parameters into one model that returns calibrated risk.

I treat this as a screening and triage problem, not just as a raw classification task. That is why the output is not limited to a single yes/no answer. My meta-calculator can assign a case to:

- `rule-out` for lower-risk patients,
- `rule-in` for higher-risk patients,
- `grey zone` for uncertain cases that should be evaluated further.

## My Contribution

I wrote the Bachelor's thesis behind this project and I also designed, implemented, tested, and iteratively improved the whole solution myself.

My work included:

- preparing and standardizing the dataset for modeling,
- deriving additional features from established medical calculators,
- benchmarking classical hepatology calculators,
- building and comparing decision tree, random forest, and logistic variants,
- calibrating predicted probabilities,
- designing threshold-based `rule-out` / `rule-in` / `grey zone` logic,

## Data

This project is based on private medical data obtained from the Clinic of Infectious Diseases in Poznan.

I cannot publish those medical data in this repository. For that reason:

- the raw and processed datasets are not part of the public repo,
- patient-level records are not available here,
- the repository is intended to expose the model, code, artifacts, and application layer without exposing the medical source data.

I used those data to train, test, and evaluate the models, but the public version of this repository is being prepared so that the final web application can work from an already trained model and no medical dataset has to be distributed with the codebase.

## Core Idea Behind the Meta-Calculator

I built the meta-calculator on top of four established non-invasive hepatology scores:

- `APRI`
- `FIB-4`
- `NAFLD Fibrosis Score`
- `de Ritis`

I do not treat those calculators as final decisions. I use them as informative features. In stronger model variants, I extend them with additional clinical and laboratory parameters such as:

- age,
- platelets,
- AST,
- ALT,
- albumin.

The model outputs a calibrated probability:

`p = P(FSCORE 3-4)`

I then convert that probability into a clinically more useful triage decision using two thresholds:

- `t_out` for low-risk screening,
- `t_in` for high-risk screening.

Values between those thresholds are deliberately left in the `grey zone` instead of forcing an overconfident binary answer.

## Why I Built a Meta-Calculator Instead of Using Existing Calculators Alone

In my thesis work, I first evaluated existing calculators on my dataset and treated them as strong baselines. That comparison showed that each individual calculator has useful properties, but also important weaknesses.

My meta-calculator improves on the current hepatology calculators in several ways:

- it integrates information from multiple calculators instead of depending on one threshold-based score,
- it produces calibrated probability rather than only a hard categorical rule,
- it gives me direct control over the safety/precision trade-off through threshold design,
- it handles uncertainty explicitly with a `grey zone`,
- it can be tuned toward screening priorities such as minimizing missed advanced fibrosis.

## Where My Meta-Calculator Outperforms Existing Calculators

Based on the comparisons from my thesis and supporting presentation materials, the main advantage of my meta-calculator is that it is safer as a screening-oriented tool for advanced fibrosis.

### Compared with FIB-4

`FIB-4` was the strongest standalone baseline in my experiments, but my meta-calculator still had important advantages in screening logic.

In my comparison against `FIB-4`:

- my meta-calculator was less likely to miss advanced fibrosis,
- my meta-calculator detected more patients with advanced fibrosis,
- my meta-calculator was better aligned with a safety-first screening strategy,
- `FIB-4` remained more selective, but that selectivity came with more missed severe cases.

From the comparison figure used in my project materials:

- missed advanced fibrosis cases were about `~5%` for my meta-calculator vs `~11%` for `FIB-4`,
- detection of advanced fibrosis cases was about `~77%` for my meta-calculator vs `~50%` for `FIB-4`.

In practice, I interpret this as:

- my meta-calculator is better when I want to reduce the risk of overlooking severe fibrosis,
- `FIB-4` is more selective, but that can come at the cost of missing more high-risk patients.

### Compared with standalone hepatology calculators overall

In my thesis-stage comparison:

- `FIB-4` was the strongest single baseline overall,
- `APRI` had useful specificity but weaker sensitivity for advanced fibrosis,
- `NAFLD Fibrosis Score` was strong for confirmation in some settings but weak for detecting advanced fibrosis,
- `de Ritis` was not reliable enough to be treated as a standalone screening tool for advanced fibrosis.

My meta-calculator improved on that landscape by combining these signals and optimizing the final decision logic around the actual screening objective.

## Selected Results

From the thesis-stage evaluation:

- among the standalone calculators, `FIB-4` was the strongest balanced baseline,
- the calibrated random-forest meta-calculator achieved `ROC AUC ~= 0.88` on the full patient cohort,
- outside the grey zone, my meta-calculator reached `Sensitivity = 0.947`, `Specificity = 0.740`, and `F1-score = 0.853` for `FSCORE 3-4`,
- on the control subset, the thesis chapter reported `ROC AUC ~= 0.8869`.

I treat these as research-stage results, not as production-grade clinical validation.

## Current Direction of the Repository

The thesis chapter documents the random-forest screening pipeline in detail, but this repository also contains later work beyond the thesis text.

Right now, my main practical candidate for further development is:

- `scripts/train_meta_logistic.py`

This reflects my current direction toward a simpler and easier-to-deploy model for the future web application.

## Repository Structure

```text
.
|-- .github/workflows/     # CI
|-- docs/                  # Project documentation
|-- legacy/                # Archived thesis-era experiments, reports, and plots
|-- models/                # Serialized models and metadata
|-- notebooks/             # Space for future notebooks
|-- references/            # Supporting notes and references
|-- reports/               # Metrics, figures, and analysis outputs
|-- scripts/               # Training and serving entrypoints
|-- src/liver_calculator/  # Reusable inference and API package
`-- tests/                 # Automated tests
```

## Quick Start

```bash
python -m venv .venv
. .venv/Scripts/activate
python -m pip install --upgrade pip
python -m pip install -e .[dev,train]
```

To train the current model candidate:

```bash
python scripts/train_meta_logistic.py
```

To run the local API scaffold:

```bash
python scripts/serve_api.py
```

Then open:

```text
http://127.0.0.1:8000
```

## Run with Docker

To run the local web app and API through Docker:

```bash
docker compose up --build
```

Then open:

```text
http://127.0.0.1:8000
```

The containerized app serves:

- the local browser interface at `/`,
- the prediction API at `/predict`,
- interactive API docs at `/docs`.

## Product Goal

I want this repository to evolve from a thesis-origin ML project into a usable web application where a user can:

1. enter laboratory and calculator-related values,
2. receive a probability estimate for `FSCORE 3-4`,
3. get a triage-oriented interpretation such as `rule-out`, `rule-in`, or `grey zone`.

The repository already contains the first FastAPI-based scaffolding for that direction.

## Limitations

- This is not a certified medical device.
- I cannot publish the medical dataset used for training and evaluation.
- External validation is limited.
- Some code in `legacy/` still reflects exploratory research work.
- Results on the small control group should be interpreted cautiously.

## Medical Disclaimer

I treat this repository as a research, educational, and portfolio project. It must not be used as a standalone clinical decision system or as a substitute for physician judgment.

## References

I structured the public version of this repository using common ML project and GitHub documentation patterns:

- GitHub repository best practices: https://docs.github.com/en/repositories/creating-and-managing-repositories/best-practices-for-repositories
- Cookiecutter Data Science: https://cookiecutter-data-science.drivendata.org/
- DrivenData Cookiecutter Data Science repository: https://github.com/drivendataorg/cookiecutter-data-science
- FastAPI full-stack template: https://github.com/fastapi/full-stack-fastapi-template
