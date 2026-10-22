.DEFAULT_GOAL := help
.PHONY: help install install-ml hooks lint format typecheck test check clean

UV ?= uv

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install: ## Create .venv (Python 3.11) with core + dev dependencies
	$(UV) sync --group dev

install-ml: ## Install the full causal-ML stack as well
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

test: ## Run the test suite
	$(UV) run pytest

check: lint typecheck test ## Everything CI runs

clean: ## Remove caches and build artefacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov build dist
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
