# Makefile for ssm-foundations.
#
# Aliases the common build + lint commands so pre-commit (audit F9), CI
# (audit F14, Track C), and local development invoke byte-identical
# commands. Modeled in spirit on post_transformers' Makefile (which is
# 215 lines; this one stays small until more targets are warranted).
#
# Audit reference: audits/2026-05-25_standards_vs_post_transformers.md#F10

.PHONY: help validate build-bib build-labels build status-snapshot status-check check-bibkeys check-xrefs lint check companion-julia-tests

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
	@echo "  status-check      - exit 1 if docs/STATUS.md is >14 days stale"
	@echo ""
	@echo "  check-bibkeys     - validate bibliography.bib + chapter citations (audit F6)"
	@echo "  check-xrefs       - validate <Theorem id=...>  + <Figure id=...> (audit F7)"
	@echo "  companion-julia-tests"
	@echo "                    - run julia runtests.jl in ch05, ch06 (ch04 manual; needs Pkg.instantiate)"
	@echo ""
	@echo "  lint              - check-bibkeys + check-xrefs (fast)"
	@echo "  check             - validate + lint + status-check (full gate)"

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

# Companion testing (audit F8 — Julia track).
#
# Runs in ch05 + ch06 (no DifferentialEquations.jl dependency, so quick).
# ch04 requires Pkg.instantiate first; we don't include it in the default
# loop. To run ch04: julia --project=companions/ch04/julia -e 'using Pkg;
# Pkg.instantiate()' first, then julia --project=companions/ch04/julia
# companions/ch04/julia/runtests.jl.

companion-julia-tests:
	@for ch in ch05 ch06; do \
		echo "==> $$ch julia tests"; \
		julia --project=companions/$$ch/julia companions/$$ch/julia/runtests.jl || exit 1; \
	done

# Companion testing (JAX track — audit 0527-F26). Local gate only: deliberately
# NOT wired into `check` (which CI runs, and where jax is unavailable). Run via
# pre-commit / `make check-local`. Requires the uv-managed .venv (see .gitignore).
companion-jax-tests:
	@.venv/bin/pytest companions -q

# Composite gates.

lint: check-bibkeys check-xrefs

check: validate lint status-check
	@echo ""
	@echo "make check: all gates passed"

# Local-only super-gate: `check` plus the companion test suites that need the uv
# .venv (jax/pytest). NOT used by CI — validate.yml runs `check`, kept jax-free.
# Run before pushing a Track-B PR.
check-local: check companion-jax-tests
	@echo "make check-local: content gates + companion tests passed"
