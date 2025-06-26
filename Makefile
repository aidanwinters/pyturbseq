.PHONY: help install install-dev test test-fast test-integration lint format clean prepare-test-data setup-dev

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install package in development mode
	pip install -e .

install-dev:  ## Install package with development dependencies
	pip install -e ".[dev,test]"

setup-dev:  ## Complete development setup (install + pre-commit)
	pip install -e ".[dev,test]"
	pre-commit install

test:  ## Run all tests
	pytest

test-fast:  ## Run fast tests only (skip slow/integration tests)
	pytest -m "not slow and not integration"

test-integration:  ## Run integration tests only
	pytest -m "integration"

test-coverage:  ## Run tests with coverage report
	pytest --cov=pyturbseq --cov-report=html --cov-report=term

test-watch:  ## Run tests in watch mode
	pytest-watch

lint:  ## Run linting checks
	flake8 pyturbseq tests
	mypy pyturbseq

format:  ## Format code with black and isort
	black pyturbseq tests
	isort pyturbseq tests

format-check:  ## Check code formatting without making changes
	black --check pyturbseq tests
	isort --check-only pyturbseq tests

pre-commit:  ## Run pre-commit on all files
	pre-commit run --all-files

prepare-test-data:  ## Prepare test datasets from real data
	python tests/prepare_test_data.py

clean:  ## Clean up build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build:  ## Build package
	python -m build

check-deps:  ## Check if dependencies can be installed with current Python
	pip install --dry-run -e ".[dev,test]"

release-test:  ## Build and upload to test PyPI
	python -m build
	twine upload --repository testpypi dist/*

release:  ## Build and upload to PyPI
	python -m build
	twine upload dist/*

docs:  ## Build documentation
	cd docs && make html

docs-serve:  ## Serve documentation locally
	cd docs/_build/html && python -m http.server 8000 