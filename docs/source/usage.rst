Usage
=====

Installation
------------

With Poetry (the group standard)::

    poetry install
    poetry shell

or with pip::

    pip install -r requirements-dev.txt
    pip install -e .

Running the pipeline
--------------------

Every step has a ``make`` target::

    make data        # clean and split
    make features    # fit the preprocessor
    make train-all   # train all candidates, register the best
    make predict     # score the test split
    make explain     # SHAP explanations (part 3)
    make ui          # open the MLflow tracking UI

or run the whole thing at once::

    make all

As an MLflow project
--------------------

The repository is also a packaged MLflow project, so it can be run without
installing anything by hand::

    mlflow run . -e main --env-manager local
    mlflow run . -e train -P model=random_forest --env-manager local

Serving
-------

The registered champion model is served on a local REST endpoint::

    make serve            # mlflow models serve, port 5001
    make score            # send a sample request

The endpoint accepts the standard MLflow ``dataframe_split`` payload and
returns one churn probability per row.

Explainability
--------------

``make explain`` builds a SHAP ``TreeExplainer`` on the registered champion and
writes every figure to ``reports/figures/shap/`` plus per-customer reason codes
to ``reports/predictions_with_reasons.csv``::

    make explain                     # highest-risk customer
    make explain CUSTOMER=54023      # a specific customer

Quality gates
-------------

::

    make format   # black
    make lint     # flake8 + pylint
    make test     # pytest with coverage
    make docs     # this documentation

The same four commands run in GitHub Actions on every push and pull request.
