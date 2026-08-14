PY ?= python
PIP ?= $(PY) -m pip

.PHONY: help install fixtures fixtures-index doctor test lint format clean verify-phase-0 verify-phase-1 verify-phase-2 verify-phase-3

help:
	@echo "install         install the package with core + dev extras"
	@echo "fixtures        regenerate committed fixtures (core-only)"
	@echo "fixtures-index  regenerate the committed index fixture (needs embed extra)"
	@echo "doctor          run the environment health check"
	@echo "test            run the test suite"
	@echo "lint            ruff check"
	@echo "verify-phase-0  full Phase 0 acceptance run"
	@echo "verify-phase-1  full Phase 1 acceptance run"
	@echo "verify-phase-2  full Phase 2 acceptance run"
	@echo "verify-phase-3  full Phase 3 acceptance run"

install:
	$(PIP) install -e ".[core,dev]"

fixtures:
	$(PY) scripts/build_fixtures.py

# Needs the `embed` extra (chromadb) -- separate from `fixtures` above so
# verify-phase-0/1/2 never need to install it.
fixtures-index:
	$(PY) scripts/build_index_fixture.py

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

# Phase 2 acceptance: all four chunkers run against fixtures with no prior
# phase run, api_docs demonstrates the markdown-vs-fixed split-code-block
# punchline (AC-5) at a chunk size small enough to force it, and tests pass.
# 128 tokens is deliberately below every api_docs doc's fence size at the
# default 512 -- see tests/test_phase_2.py for the token-count survey this
# was chosen from.
verify-phase-2:
	@echo "==> installing (core + dev only)"
	@$(PIP) install -e ".[core,dev]" -q
	@echo "==> checking fixture determinism"
	@$(PY) scripts/build_fixtures.py --check
	@echo "==> building all corpora"
	@$(PY) -m rag_lab.cli corpus build --all
	@echo "==> chunking api_docs: fixed vs markdown at a size that forces a fence split"
	@$(PY) -m rag_lab.cli chunk run --corpus api_docs --chunker fixed --params chunk_tokens=128 --params overlap_tokens=32
	@$(PY) -m rag_lab.cli chunk run --corpus api_docs --chunker markdown --params max_tokens=128
	@echo "==> doctor"
	@$(PY) -m rag_lab.cli doctor
	@echo "==> tests"
	@$(PY) -m pytest tests/test_phase_0.py tests/test_phase_1.py tests/test_phase_2.py -q
	@echo "PHASE 2 OK"

# Phase 3 acceptance: builds an index from a real chunk set with the actual
# BGE model (AC-1/2/3/4/6 all need the real, trained embedder -- a fake one
# can't make a meaningful truncation-vs-recall claim), and a repeat build
# hits the cache (AC-5). Also re-checks the two fixture generators, since
# both are consumed by later phases' independence-contract runs.
verify-phase-3:
	@echo "==> installing (core + dev + embed)"
	@$(PIP) install -e ".[core,dev,embed]" -q
	@echo "==> checking fixture determinism (core fixtures + index fixture)"
	@$(PY) scripts/build_fixtures.py --check
	@$(PY) scripts/build_index_fixture.py --check
	@echo "==> building api_docs and a chunk set to index"
	@$(PY) -m rag_lab.cli corpus build --all
	@CHUNK_SET_ID=$$($(PY) -m rag_lab.cli chunk run --corpus api_docs --chunker recursive | grep -oE 'api_docs__recursive__[a-f0-9]+'); \
	echo "==> building a BGE index (AC-1) and re-running for the cache hit (AC-5)"; \
	$(PY) -m rag_lab.cli index build --chunk-set $$CHUNK_SET_ID --embedder bge-small; \
	$(PY) -m rag_lab.cli index build --chunk-set $$CHUNK_SET_ID --embedder bge-small
	@echo "==> doctor"
	@$(PY) -m rag_lab.cli doctor
	@echo "==> tests"
	@$(PY) -m pytest tests/test_phase_0.py tests/test_phase_1.py tests/test_phase_2.py tests/test_phase_3.py -q
	@echo "PHASE 3 OK"
