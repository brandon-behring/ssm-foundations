# Release-trust dashboard — ssm-foundations

At-a-glance scoreboard of per-chapter artifact completeness and verification state.
The at-a-glance complement to the deeper audit capability (see
`audits/2026-06-04_ecosystem_checkpoint.md` §4). Hand-maintained; refresh the
artifact columns with the snippet at the bottom and reconcile `status:` by eye.

**Verified:** 2026-06-13 (post Ch-15 ship: counter-evidence and diagnostic tools authored
vertical — JAX + torch + a stdlib Julia QR-Lyapunov cross-check, three figures, four review
subagents; the prosecution's file — cited impossibility theory (TC⁰, the illusion of state,
the copying separation) plus an architecture-agnostic capacity bound and
Lyapunov/effective-state-size diagnostics validated on constructed systems, the instruments
pilot B will run on trained models).

## Scoreboard

| Ch | Title | Part | Status | JAX | Julia | torch | Tests | Figs | Live |
|----|-------|------|--------|:---:|:-----:|:-----:|:-----:|:----:|:----:|
| 1  | Linear ODEs as state-space systems | foundations | `implemented` | ✓ | — | ✓ | ✓ | 3 | ✓ |
| 2  | Stability theory: Lyapunov, A-stability, BIBO | foundations | `implemented` | ✓ | — | ✓ | ✓ | 2 | ✓ |
| 3  | Linear algebra: structured matrices and conditioning | foundations | `implemented` | ✓ | — | ✓ | ✓ | 2 | ✓ |
| 4  | Discretization: ZOH, bilinear, exponential families | foundations | `implemented` | ✓ | ✓ | ✓ | ✓ | 3 | ✓ |
| 5  | Stability regions, Butcher tableau, order conditions | foundations | `implemented` | ✓ | ✓ | ✓ | ✓ | 4 | ✓ |
| 6  | Implicit methods, stiff systems, symplectic integration | foundations | `implemented` | ✓ | ✓ | ✓ | ✓ | 3 | ✓ |
| 7  | HiPPO theory: orthogonal-basis projection operators | ssm-core | `implemented` | ✓ | ✓ | ✓ | ✓ | 4 | ✓ |
| 8  | LTI SSMs: S4, S4D, S5 | ssm-core | `implemented` | ✓ | — | ✓ | ✓ | 4 | ✓ |
| 9  | Selective SSMs: Mamba-1, Mamba-2, SSD | ssm-core | `implemented` | ✓ | — | ✓ | ✓ | 4 | ✓ |
| 10 | Mamba-3 and the exp-trapezoidal integrator | ssm-core | `implemented` | ✓ | ✓ | ✓ | ✓ | 3 | ✓ |
| 11 | Linear attention and Hyena | beyond-ssm | `implemented` | ✓ | ✓ | ✓ | ✓ | 4 | ✓ |
| 12 | Delta-rule lineage: DeltaNet, Gated DeltaNet, Kimi | beyond-ssm | `implemented` | ✓ | ✓ | ✓ | ✓ | 4 | ✓ |
| 13 | Exponential gates and matrix memory: xLSTM, RWKV-7 | beyond-ssm | `implemented` | ✓ | ✓ | ✓ | ✓ | 3 | ✓ |
| 14 | Hybrid architectures and gating mechanisms | integration | `implemented` | ✓ | — | ✓ | ✓ | 4 | ✓ |
| 15 | Counter-evidence and diagnostic tools | integration | `implemented` | ✓ | ✓ | ✓ | ✓ | 3 | ✓ |
| 16 | Empirical methodology: benchmark protocols | integration | `implemented` | ✓ | — | ✓ | ✓ | 4 | ✓ |
| 17 | Niche-pilot integration | synthesis | `planned` | — | — | — | — | 0 | stub |

**Legend.** ✓ present/passing · — absent · `Tests` = JAX *and* torch pytest suites present and green
(`make companion-jax-tests` / `companion-torch-tests`) · `Figs` = committed PNGs under `public/figures/chNN/`
· `Live` = chapter content deployed at <https://ssm-foundations.brandon-behring.dev> (`stub` = PreReleaseBanner only).

## Trust notes

- **Ch 1–16** are `implemented`: prose + exercises + companions authored, `make check` (content gates +
  bibkey/xref lint) green, companion suites green, deployed. JAX is the canonical reference for every
  chapter; **torch parity is complete for ch01–16** (ch01–10 backfilled via the Ch-11 runway, closing
  0527-F27; ch11–16 authored with parity from the start).
- **Julia is a selective track, not a universal gate** — present where the chapter's numerical core
  warrants a cross-language check (ch04 via `DifferentialEquations.jl`; ch05–07, ch10–13, and ch15 stdlib-only —
  ch13's cross-checks the exponential-gate stabilizer's log-domain overflow control, ch15's is an independent
  QR-Lyapunov + eigendecomposition cross-check of the diagnostic); absent by design elsewhere (ch14 and ch16
  are filtering linear algebra + counting — JAX+torch only, per their briefs). A `—` in the Julia column is *not* a gap.
- **`claim-skeptic` is new (2026-06-04)** and has **not** been run retroactively on Ch 1–10; its adversarial
  math-claim pass is a forward gate (exercised on Ch 11–16; on Ch 15 it confirmed all three propositions —
  the capacity bound's non-duplication of ch11/ch16, the Lyapunov estimator's coupling-not-degeneracy
  resolution limit, the two-route regime cross-check — and verified the cited-not-proven boundary for the
  TC⁰/illusion-of-state results; on Ch 13 it verified all three propositions against the authoritative NXAI
  xLSTM reference implementation, and on Ch 16 it confirmed all four propositions and drove the
  expected-vs-realized excess-CE distinction), so the Ch 1–10 rows do not yet reflect a claim-skeptic review.
- **Ch 17** is a `planned` stub (no original prose/companions); the deployed page shows only the
  pre-release banner.

## Refresh the artifact columns

```bash
for c in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17; do
  mdx=$(ls src/content/chapters/ch${c}-*.mdx 2>/dev/null | head -1); [ -z "$mdx" ] && continue
  status=$(grep -m1 '^status:' "$mdx" | sed 's/status: *//')
  jax=$([ -n "$(ls companions/ch${c}/jax/*.py 2>/dev/null)" ] && echo Y || echo -)
  jul=$([ -n "$(ls companions/ch${c}/julia/*.jl 2>/dev/null)" ] && echo Y || echo -)
  tor=$([ -n "$(ls companions/ch${c}/torch/*.py 2>/dev/null)" ] && echo Y || echo -)
  figs=$(ls public/figures/ch${c}/*.png 2>/dev/null | wc -l | tr -d ' ')
  printf 'ch%s status=%-12s jax=%s jul=%s tor=%s figs=%s\n' "$c" "$status" "$jax" "$jul" "$tor" "$figs"
done
```

Verification commands behind the columns: `make check` (CI gate), `make companion-jax-tests`,
`make companion-torch-tests`, `make companion-julia-tests` (local-only; need the uv `.venv`).
