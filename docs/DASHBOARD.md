# Release-trust dashboard — ssm-foundations

At-a-glance scoreboard of per-chapter artifact completeness and verification state.
The at-a-glance complement to the deeper audit capability (see
`audits/2026-06-04_ecosystem_checkpoint.md` §4). Hand-maintained; refresh the
artifact columns with the snippet at the bottom and reconcile `status:` by eye.

**Verified:** 2026-06-13 (post Ch-17 ship — **the book is content-complete (M6)**: the synthesis
crown-jewel authored vertical — JAX + a stdlib Julia symplectic cross-check (no torch by design),
two figures, four review subagents; two integration demos composing the book's instruments — a
symplectic *atlas cell* (Ch 6 + 10) and a two-timescale *disentanglement pipeline* (Ch 14 + 15 + 16)
— run on idealized systems as the reproducible templates the C1/B pilots fill with trained-model
data. All 17 chapters now `implemented` + deployed).

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
| 17 | Niche-pilot integration | synthesis | `implemented` | ✓ | ✓ | — | ✓ | 2 | ✓ |

**Legend.** ✓ present/passing · — absent · `Tests` = the chapter's companion suites present and green
(`make companion-jax-tests` / `companion-torch-tests` / `companion-julia-tests`, as present) · `Figs` = committed PNGs under `public/figures/chNN/`
· `Live` = chapter content deployed at <https://ssm-foundations.brandon-behring.dev> (`stub` = PreReleaseBanner only).

## Trust notes

- **Ch 1–17** are `implemented`: prose + exercises + companions authored, `make check` (content gates +
  bibkey/xref lint) green, companion suites green, deployed. JAX is the canonical reference for every
  chapter; **torch parity is complete for ch01–16** (ch01–10 backfilled via the Ch-11 runway, closing
  0527-F27; ch11–16 authored with parity from the start). **Ch 17 ships JAX + Julia only — no torch by
  design** (a synthesis chapter that *composes* existing JAX instruments introduces no new kernel to port).
- **Julia is a selective track, not a universal gate** — present where the chapter's numerical core
  warrants a cross-language check (ch04 via `DifferentialEquations.jl`; ch05–07, ch10–13, ch15, and ch17 stdlib-only —
  ch13's cross-checks the exponential-gate stabilizer, ch15's the QR-Lyapunov diagnostic, ch17's the C1 symplectic
  atlas cell's energy conservation, the C1 pilot's atlas being itself Julia); absent by design elsewhere (ch14 and ch16
  are filtering linear algebra + counting — JAX+torch only, per their briefs). A `—` in the Julia column is *not* a gap.
- **`claim-skeptic` is new (2026-06-04)** and has **not** been run retroactively on Ch 1–10; its adversarial
  math-claim pass is a forward gate (exercised on Ch 11–17; on Ch 17 it confirmed the two integration
  signatures are genuinely new — not re-runs of the chapters they compose — the no-trained-results scope,
  and the provisional verdict bounded by the TC⁰ ceiling; on Ch 15 it confirmed all three propositions —
  the capacity bound's non-duplication of ch11/ch16, the Lyapunov estimator's coupling-not-degeneracy
  resolution limit, the two-route regime cross-check — and verified the cited-not-proven boundary for the
  TC⁰/illusion-of-state results; on Ch 13 it verified all three propositions against the authoritative NXAI
  xLSTM reference implementation, and on Ch 16 it confirmed all four propositions and drove the
  expected-vs-realized excess-CE distinction), so the Ch 1–10 rows do not yet reflect a claim-skeptic review.
- **All 17 chapters are `implemented` and deployed — the book is content-complete (M6).** Post-M6 work is
  upkeep: the retroactive `claim-skeptic` sweep over Ch 1–10, a STYLE §8 companion-section refresh, and the
  toolkit re-bumps tracked in the repo's issues.

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
