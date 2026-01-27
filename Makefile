# =============================================================================
# Probabilistic Generative Model - Makefile
# =============================================================================
#
# Usage:
#   make help        Show available commands
#   make install     Install dependencies
#   make init-db     Initialize database
#   make ingest      Run daily ingestion
#   make test        Run tests
#   make lint        Run linters
#

.PHONY: help install init-db ingest test lint format clean

# Default target
help:
	@echo "Available commands:"
	@echo "  make install     Install dependencies"
	@echo "  make init-db     Initialize database schema"
	@echo "  make drop-db     Drop and recreate database (DESTRUCTIVE)"
	@echo "  make ingest      Run daily ingestion (yesterday + today)"
	@echo "  make ingest-season SEASON=2024  Ingest full season"
	@echo "  make coverage    Show HT coverage report"
	@echo "  make quality     Run data quality checks"
	@echo "  make test        Run tests"
	@echo "  make test-cov    Run tests with coverage"
	@echo "  make lint        Run linters"
	@echo "  make format      Format code"
	@echo "  make clean       Clean cache and build files"

# =============================================================================
# Setup
# =============================================================================

install:
	pip install -e ".[dev,notebooks]"

install-dev:
	pip install -e ".[dev]"

# =============================================================================
# Database
# =============================================================================

init-db:
	python scripts/init_db.py

init-db-dry:
	python scripts/init_db.py --dry-run

drop-db:
	python scripts/init_db.py --drop

verify-db:
	python scripts/init_db.py --verify

# =============================================================================
# Data Ingestion
# =============================================================================

ingest:
	python scripts/daily_ingest.py --quality

ingest-season:
	python scripts/daily_ingest.py --season $(SEASON) --leagues PL BL1 SA PD FL1

coverage:
	python scripts/daily_ingest.py --coverage

quality:
	python scripts/daily_ingest.py --quality

# =============================================================================
# Testing
# =============================================================================

test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html

test-fast:
	pytest tests/ -v -x --ff

# =============================================================================
# Code Quality
# =============================================================================

lint:
	ruff check src/ tests/
	mypy src/ --ignore-missing-imports

format:
	black src/ tests/ scripts/
	ruff check src/ tests/ --fix

# =============================================================================
# Cleanup
# =============================================================================

clean:
	rm -rf .cache/
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

# =============================================================================
# Development
# =============================================================================

api:
	uvicorn src.api.main:app --reload --port 8000

dashboard:
	streamlit run src/dashboard/app.py

notebook:
	jupyter lab notebooks/
