PY ?= python
PIP ?= $(PY) -m pip

.PHONY: help install fixtures doctor test lint format clean verify-phase-0 verify-phase-1

help:
	@echo "install         install the package with core + dev extras"
	@echo "fixtures        regenerate committed fixtures"
	@echo "doctor          run the environment health check"
	@echo "test            run the test suite"
	@echo "lint            ruff check"
	@echo "verify-phase-0  full Phase 0 acceptance run"
	@echo "verify-phase-1  full Phase 1 acceptance run"

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

# Phase 1 acceptance: corpora build cleanly, stats show the expected signal,
# tests pass. Re-checks fixture determinism too, so this target is safe to
# run in isolation on a clean checkout without verify-phase-0 having run first.
verify-phase-1:
	@echo "==> installing (core + dev only)"
	@$(PIP) install -e ".[core,dev]" -q
	@echo "==> checking fixture determinism"
	@$(PY) scripts/build_fixtures.py --check
	@echo "==> building all corpora"
	@$(PY) -m rag_lab.cli corpus build --all
	@echo "==> corpus stats"
	@$(PY) -m rag_lab.cli corpus stats
	@echo "==> doctor"
	@$(PY) -m rag_lab.cli doctor
	@echo "==> tests"
	@$(PY) -m pytest tests/test_phase_0.py tests/test_phase_1.py -q
	@echo "PHASE 1 OK"
