.PHONY: help install test lint format clean docs

help:
	@echo "JAX-CTSM Development Commands"
	@echo "=============================="
	@echo "make install      - Install package in editable mode with dev dependencies"
	@echo "make test         - Run all tests with coverage"
	@echo "make test-fast    - Run tests without coverage"
	@echo "make lint         - Run linting checks (ruff, mypy)"
	@echo "make format       - Format code with black"
	@echo "make clean        - Remove build artifacts and cache"
	@echo "make docs         - Build documentation"
	@echo "make example      - Run basic maintenance respiration example"

install:
	pip install -e ".[dev]"

test:
	pytest tests/ --cov=src/jax_ctsm --cov-report=term-missing --cov-report=html -v

test-fast:
	pytest tests/ -v

lint:
	@echo "Running ruff..."
	ruff check src/ tests/
	@echo "Running mypy..."
	mypy src/jax_ctsm

format:
	@echo "Formatting with black..."
	black src/ tests/ examples/
	@echo "Sorting imports..."
	ruff check --select I --fix src/ tests/ examples/

clean:
	@echo "Cleaning build artifacts..."
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .mypy_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	@echo "Cleaned!"

docs:
	@echo "Building documentation..."
	cd docs && make html

example:
	python examples/basic_maintenance_respiration.py

# Pre-commit style checks
check: format lint test-fast
	@echo "All checks passed!"
