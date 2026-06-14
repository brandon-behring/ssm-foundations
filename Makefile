# Makefile for ssm-foundations.
#
# Aliases the common build + lint commands so pre-commit (audit F9), CI
# (audit F14, Track C), and local development invoke byte-identical
# commands. Modeled in spirit on post_transformers' Makefile (which is
# 215 lines; this one stays small until more targets are warranted).
#
# Audit reference: audits/2026-05-25_standards_vs_post_transformers.md#F10

.PHONY: help validate build-bib build-labels build status-snapshot status-check check-bibkeys check-xrefs test-scripts lint check companion-julia-instantiate companion-julia-tests companion-jax-tests companion-torch-tests check-local check-local-torch

# Default — print available targets.
help:
	@echo "ssm-foundations Makefile targets"
	@echo ""
	@echo "  validate          - npm run validate (book-scaffold content validation)"
	@echo "  build-bib         - regenerate src/data/references.json from bibliography.bib"
	@echo "  build-labels      - regenerate src/data/labels.json from chapter id= attributes"
	@echo "  build             - full Astro build (chained via package.json prebuild hook)"
	@echo ""
	@echo "  status-snapshot   - regenerate docs/STATUS.md"
	@echo "  status-check      - exit 1 if docs/STATUS.md content drifted or is >14 days stale"
	@echo ""
	@echo "  check-bibkeys     - validate bibliography.bib + chapter citations (audit F6)"
	@echo "  check-xrefs       - validate <Theorem id=...>  + <Figure id=...> (audit F7)"
	@echo "  companion-julia-instantiate"
	@echo "                    - one-time Pkg.instantiate for ch04 (DifferentialEquations.jl)"
	@echo "  companion-julia-tests"
	@echo "                    - run julia runtests.jl (ch04/05/06/07/10/11/12/13/15/17; ch04 needs companion-julia-instantiate once)"
	@echo "  companion-jax-tests   - run JAX companion pytest suites (.venv; excludes torch)"
	@echo "  companion-torch-tests - run PyTorch companion pytest suites (.venv [torch] extra)"
	@echo ""
	@echo "  test-scripts      - node --test for scripts/ (audit #26 status-check content guard)"
	@echo "  lint              - check-bibkeys + check-xrefs (fast)"
	@echo "  check             - validate + lint + test-scripts + status-check (full gate)"

# Build / content pipeline.
#
# Note: `validate` depends on build-bib + build-labels because
# book-scaffold validate resolves <Cite key="..."> and id="..." against
# src/data/references.json and src/data/labels.json, both of which are
# gitignored (regenerated from bibliography.bib + chapter MDX). On a
# fresh checkout these files don't exist, so validate must regenerate
# them first — locally this is usually a no-op, but in CI it's the
# difference between green and 25 "unknown bibkey" errors.

build-bib:
	npm run build:bib

build-labels:
	npm run build:labels

validate: build-bib build-labels
	npm run validate

build:
	npm run build

# Status snapshot (audit F11).

status-snapshot:
	node scripts/generate-status.mjs

status-check:
	node scripts/generate-status.mjs --check

# Lint scripts (audit F6 + F7).

check-bibkeys:
	node scripts/check-bibkeys.mjs

check-xrefs:
	node scripts/check-xref-labels.mjs

# Script unit tests (audit #26 — exercises the content-validating status-check:
# committed docs/STATUS.md must be a faithful render of src/content/chapters/).
# node-only, so it runs in CI inside `make check` (no jax/torch/julia needed).
test-scripts:
	node --test scripts/*.test.mjs

# Companion testing (audit F8 — Julia track; audit #4/F7 folded ch04 in).
#
# ch05–ch17 use only stdlib (LinearAlgebra etc.), so they run with no setup.
# ch04 additionally pulls DifferentialEquations.jl (the Tsit5 reference solve),
# so on a fresh checkout it needs a one-time `make companion-julia-instantiate`
# before this loop will pass. The committed Manifest.toml pins the full
# dependency tree, so that step is reproducible; after it, ch04 runs in the
# loop like any other chapter.

companion-julia-instantiate:
	julia --project=companions/ch04/julia -e 'using Pkg; Pkg.instantiate()'

companion-julia-tests:
	@for ch in ch04 ch05 ch06 ch07 ch10 ch11 ch12 ch13 ch15 ch17; do \
		echo "==> $$ch julia tests"; \
		julia --project=companions/$$ch/julia companions/$$ch/julia/runtests.jl || exit 1; \
	done

# Companion testing (JAX track — audit 0527-F26). Local gate only: deliberately
# NOT wired into `check` (which CI runs, and where jax is unavailable). Run via
# pre-commit / `make check-local`. Requires the uv-managed .venv (see .gitignore).
companion-jax-tests:
	@.venv/bin/pytest companions -q --ignore-glob='*/torch/*'

# Companion testing (PyTorch track — audit 0527-F26). Separate target because
# torch is an optional, heavy dependency: install it with
#   uv pip install -e 'companions/_shared[torch]'
# Local gate only; never wired into `check` (CI stays jax/torch-free).
companion-torch-tests:
	@.venv/bin/pytest companions/ch01/torch companions/ch02/torch companions/ch03/torch companions/ch04/torch companions/ch05/torch companions/ch06/torch companions/ch07/torch companions/ch08/torch companions/ch09/torch companions/ch10/torch companions/ch11/torch companions/ch12/torch companions/ch13/torch companions/ch14/torch companions/ch15/torch companions/ch16/torch -q

# Composite gates.

lint: check-bibkeys check-xrefs

check: validate lint test-scripts status-check
	@echo ""
	@echo "make check: all gates passed"

# Local-only super-gate: `check` plus the companion test suites that need the uv
# .venv (jax/pytest). NOT used by CI — validate.yml runs `check`, kept jax-free.
# Run before pushing a Track-B PR.
check-local: check companion-jax-tests
	@echo "make check-local: content gates + companion tests passed"

# Full local super-gate including the PyTorch companions (needs the [torch] extra).
# CI still runs only `check`; this is the broadest local gate.
check-local-torch: check companion-jax-tests companion-torch-tests
	@echo "make check-local-torch: content gates + jax + torch companion tests passed"
