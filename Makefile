.PHONY: help install data features train train-all predict sweep relocate ui serve score docs lint format test all clean

PYTHON ?= python
PORT   ?= 5001

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install:  ## Install the project and its dev dependencies
	$(PYTHON) -m pip install -r requirements-dev.txt
	$(PYTHON) -m pip install -e .

data:  ## Clean the raw file and build the train/test split
	$(PYTHON) -m churn.data.make_dataset

features:  ## Fit and save the preprocessing pipeline
	$(PYTHON) -m churn.features.build_features

train:  ## Train a single model (MODEL=lightgbm)
	$(PYTHON) -m churn.models.train_model --model $(or $(MODEL),lightgbm)

train-all:  ## Train every candidate plus the LightGBM sweep, register the best
	$(PYTHON) -m churn.models.train_model --all --sweep --register

sweep:  ## Run only the LightGBM hyper-parameter sweep
	$(PYTHON) -m churn.models.train_model --sweep

predict:  ## Score the test split with the champion model
	$(PYTHON) -m churn.models.predict_model

relocate:  ## Point the committed MLflow store at this checkout
	$(PYTHON) scripts/relocate_mlflow.py

ui: relocate  ## Open the local MLflow tracking UI
	mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000 --workers 1

serve: relocate  ## Serve the champion model on a local REST endpoint
	MLFLOW_TRACKING_URI=sqlite:///mlflow.db \
	mlflow models serve -m "models:/ecommerce-churn-classifier@champion" -p $(PORT) --env-manager local

score:  ## Send a sample request to the running endpoint
	$(PYTHON) scripts/score_request.py --port $(PORT)

docs:  ## Build the Sphinx HTML documentation
	sphinx-build -b html docs/source docs/build

lint:  ## Run flake8 and pylint
	flake8 src tests
	pylint src/churn

format:  ## Reformat with black
	black src tests scripts

test:  ## Run the test suite
	pytest --cov=churn --cov-report=term-missing

all: data features train-all predict  ## Run the whole pipeline end to end

clean:  ## Remove generated data, models and docs
	rm -rf data/interim/* data/processed/* models/*.joblib docs/build mlflow.db mlruns
