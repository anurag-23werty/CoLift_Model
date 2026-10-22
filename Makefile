.DEFAULT_GOAL := help
.PHONY: help install hooks lint format typecheck test test-all check clean data data-download validate split eda

UV ?= uv

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install: ## Create .venv (Python 3.11) with core, causal-ML and dev dependencies
	$(UV) sync --group dev --extra ml

hooks: ## Install git pre-commit hooks
	$(UV) run pre-commit install

lint: ## Ruff lint + format check
	$(UV) run ruff check .
	$(UV) run ruff format --check .

format: ## Auto-fix lint issues and format code
	$(UV) run ruff check --fix .
	$(UV) run ruff format .

typecheck: ## Static type checking
	$(UV) run mypy

test: ## Run fast tests (what CI runs)
	$(UV) run pytest -m "not slow"

test-all: ## Run every test, including full-Criteo checks
	$(UV) run pytest

check: lint typecheck test ## Everything CI runs

data: ## Full data pipeline via DVC (download -> validate -> split), only stale stages rerun
	$(UV) run dvc repro

data-download: ## Download Hillstrom + full Criteo (~311 MB) to data/raw
	$(UV) run python -m uplift.data.download --dataset all

validate: ## Schema, SRM, leakage and balance checks -> reports/validation/
	$(UV) run python -m uplift.data.validate --dataset all

split: ## Stratified, hash-verified train/val/test -> data/processed/
	$(UV) run python -m uplift.data.split --dataset all

eda: ## Re-execute the EDA notebook in place and refresh reports/figures/
	$(UV) run jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb

clean: ## Remove caches and build artefacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov build dist
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
