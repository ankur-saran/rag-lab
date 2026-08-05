PY ?= python
PIP ?= $(PY) -m pip

.PHONY: help install fixtures doctor test lint format clean verify-phase-0

help:
	@echo "install         install the package with core + dev extras"
	@echo "fixtures        regenerate committed fixtures"
	@echo "doctor          run the environment health check"
	@echo "test            run the test suite"
	@echo "lint            ruff check"
	@echo "verify-phase-0  full Phase 0 acceptance run"

install:
	$(PIP) install -e ".[core,dev]"

fixtures:
	$(PY) scripts/build_fixtures.py

doctor:
	$(PY) -m rag_lab.cli doctor

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check src tests scripts

format:
	$(PY) -m ruff format src tests scripts

clean:
	rm -rf artifacts/documents/* artifacts/chunks/* artifacts/indexes/* \
	       artifacts/evalset/* artifacts/results/* .pytest_cache .ruff_cache
	find . -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true

# Phase 0 acceptance: clean install, deterministic fixtures, healthy env, tests pass.
verify-phase-0:
	@echo "==> installing (core + dev only)"
	@$(PIP) install -e ".[core,dev]" -q
	@echo "==> checking fixture determinism"
	@$(PY) scripts/build_fixtures.py --check
	@echo "==> doctor"
	@$(PY) -m rag_lab.cli doctor
	@echo "==> tests"
	@$(PY) -m pytest tests/test_phase_0.py -q
	@echo "PHASE 0 OK"
