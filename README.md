# E-Commerce Customer Churn Prediction

[![CI](https://github.com/ribbzz/AI_Project_Methodology/actions/workflows/ci.yml/badge.svg)](https://github.com/ribbzz/AI_Project_Methodology/actions/workflows/ci.yml)

EPITA International Programs - **AI Project Methodology 2026**, graded project
parts 2 (technical implementation) and 3 (explainable AI).
Group: **Rabeeh Abou Ismail, Amjad Bsat, Majd Hammoud**.

Part 1 of the project framed a churn prediction initiative for the fictional
marketplace *RetailGenius*. This repository implements it as a production-style
ML project: separate pipeline steps, a standard project layout, PEP8 tooling,
reproducible environments, generated documentation, MLflow for tracking,
packaging, model registry and serving, and SHAP for explainability.

> Model performance is not the goal of this project (see the brief). The goal is
> the engineering practice around the model.

**Reports:** [Part 1 - functional framing](reports/AI_PM_Graded_Project_Part1.pdf) ·
[Parts 2 and 3 - technical implementation and explainable AI](reports/AI_PM_Graded_Project_Part2.pdf)

## Dataset

[E-Commerce Customer Churn Analysis and Prediction](https://www.kaggle.com/datasets/ankitverma2010/ecommerce-customer-churn-analysis-and-prediction)
(Kaggle) - 5,630 customers, 20 columns, **16.8% churners**. The raw workbook is
committed in `data/raw/` so that the pipeline runs without a Kaggle account.

## Project structure

Adapted from [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/).

```
├── data
│   ├── raw/                 original, immutable workbook
│   ├── interim/             cleaned table            (generated, git-ignored)
│   └── processed/           train / test split       (generated, git-ignored)
├── docs/source/             Sphinx sources (make docs)
├── models/                  fitted preprocessing pipeline
├── mlflow.db, mlruns/       MLflow tracking store, artefacts and model registry
├── notebooks/               exploratory analysis
├── reports/                 figures, model comparison, predictions
├── scripts/                 serving client, figure builder, store relocation
├── src/churn
│   ├── config.py            paths and constants in one place
│   ├── data/make_dataset.py        step 1 - data preparation
│   ├── features/build_features.py  step 2 - feature engineering
│   ├── models/train_model.py       step 3 - training, tracking, registry
│   ├── models/predict_model.py     step 4 - batch inference
│   ├── models/explain.py           step 5 - SHAP explainability (part 3)
│   ├── models/evaluate.py          metrics (PR-AUC, lift@10%, ...)
│   └── visualization/visualize.py  evaluation plots
├── tests/                   pytest suite
├── MLproject, python_env.yaml      MLflow project definition
├── pyproject.toml           Poetry project (dependencies, tooling config)
├── requirements*.txt        pip equivalents
└── Makefile                 one command per step
```

## Setup

The group standard is **Python 3.11 or newer**. Developed and tested locally on 3.14; CI runs on 3.11.

```bash
# Poetry (recommended)
poetry install
poetry shell

# or plain pip
python -m venv .venv && source .venv/bin/activate
make install
```

## Running the pipeline

```bash
make data        # 1. clean the raw file, stratified 80/20 split
make features    # 2. fit the preprocessing pipeline
make train-all   # 3. train 3 model families + a 6-run LightGBM sweep, register the best
make predict     # 4. score the test split with the registered champion
make explain     # 5. SHAP explanations of the champion (part 3)
make all         # 1 -> 5 in one go
```

Or as an MLflow project, without installing the package by hand:

```bash
mlflow run . -e main --env-manager local
mlflow run . -e train -P model=random_forest --env-manager local
```

## MLflow

```bash
make ui          # tracking UI on http://127.0.0.1:5000
make serve       # REST endpoint for the champion model on port 5001
make score       # send 6 test customers to the endpoint
```

`mlflow.db` and `mlruns/` are committed so the tracked runs and the registered
model can be inspected straight after cloning. MLflow stores absolute paths;
`make ui` and `make serve` first run `scripts/relocate_mlflow.py`, which rewrites
them to the location of your checkout.

| What | Where |
|---|---|
| Experiment | `ecommerce-churn` - 9 training runs + 1 sweep parent |
| Registered model | `ecommerce-churn-classifier`, alias `champion` |
| Served output | `[p(stay), p(churn)]` per customer (`predict_proba`) |

### Results (test split, 1,126 customers)

| Model | PR-AUC | ROC-AUC | Lift @10% | Brier |
|---|---|---|---|---|
| **LightGBM lr=0.1, leaves=31 (champion)** | **0.999** | **1.000** | **5.93** | **0.004** |
| LightGBM (defaults) | 0.999 | 1.000 | 5.93 | 0.004 |
| Random forest | 0.965 | 0.993 | 5.87 | 0.037 |
| Logistic regression | 0.724 | 0.903 | 4.55 | 0.118 |

Lift @10% cannot exceed 1 / 0.169 = 5.93 on this split, because the top 10% of
the ranking is smaller than the churner population. These scores are optimistic:
the dataset is a single static snapshot, so the out-of-time validation described
in part 1 is not possible here.

## Explainability (part 3)

```bash
make explain                  # explains the highest-risk customer
make explain CUSTOMER=54023   # explains a specific customer
```

A SHAP `TreeExplainer` computes exact Shapley values for the champion LightGBM
model. Outputs land in `reports/figures/shap/`:

| Output | File |
|---|---|
| One customer: waterfall and force plot | `waterfall_customer_*.png`, `force_customer_*.png` |
| All customers at once (interactive) | `force_all_customers.html` |
| Summary plot per class | `summary_class_churn.png`, `summary_class_no_churn.png` |
| Mean \|SHAP\| and beeswarm | `mean_shap_bar.png`, `beeswarm.png` |
| Dependence plots | `dependence_Tenure.png`, `_CashbackAmount`, `_Complain`, `_DaySinceLastOrder` |

Per-customer reason codes go to `reports/predictions_with_reasons.csv`, in the
form the Part 1 framing promised marketing:

```
CustomerID  churn_probability  risk_band  reason_1
     52343                1.0       HIGH  Complain +5.78 (raises risk)
```

Plots display features in their original units (`1 = Complain`), while the
Shapley values are computed on the scaled matrix the model actually sees.

## Quality

```bash
make format      # black (line length 100)
make lint        # flake8 + pylint
make test        # pytest + coverage
make docs        # Sphinx HTML in docs/build
```

GitHub Actions runs formatting, linting, tests and the documentation build on
every push (`.github/workflows/ci.yml`).
