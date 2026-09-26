E-Commerce Churn Prediction
===========================

Customer churn prediction for an e-commerce marketplace, built for part 2 of
the EPITA *AI Project Methodology* graded project. The functional framing that
this code implements is part 1 of the same project.

The pipeline is split into four steps, each of which is a separate module and a
separate command:

.. list-table::
   :header-rows: 1
   :widths: 22 30 48

   * - Step
     - Command
     - What it does
   * - Data preparation
     - ``python -m churn.data.make_dataset``
     - Reads the raw Excel export, merges duplicate category spellings, drops
       duplicate customers and writes a stratified train/test split.
   * - Feature engineering
     - ``python -m churn.features.build_features``
     - Adds the behavioural ratios and fits the imputation, scaling and
       one-hot encoding pipeline.
   * - Training
     - ``python -m churn.models.train_model --all --register``
     - Trains every candidate model, logs parameters, metrics and plots to
       MLflow, and registers the best run.
   * - Inference
     - ``python -m churn.models.predict_model``
     - Scores customers with the champion model and assigns a risk band.
   * - Explainability
     - ``python -m churn.models.explain``
     - Computes Shapley values with a SHAP TreeExplainer and writes the
       global, per-class and per-customer explanations.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   usage
   api

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
