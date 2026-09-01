.PHONY: help install verify lint format test clean

PYTHON = python
PIP = pip

help:
	@echo "Bi-ISL Development Tasks:"
	@echo "  make install     - Install all package dependencies"
	@echo "  make verify      - Run environment verification script"
	@echo "  make lint        - Run ruff and mypy code checks"
	@echo "  make format      - Format source code with ruff"
	@echo "  make test        - Run pytest test suite"
	@echo "  make clean       - Clean temporary build and cache files"

install:
	$(PIP) install -e .[core,vision,training,evaluation,development,deployment,optionalresearch]

verify:
	$(PYTHON) scripts/verify_environment.py

lint:
	ruff check src/ tests/
	mypy src/

format:
	ruff format src/ tests/

test:
	pytest tests/

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache *.egg-info build dist
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
