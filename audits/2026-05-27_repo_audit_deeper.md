# ssm-foundations — Deeper Repo Audit (Content, Correctness, Code Quality, Organization)

**Audit date:** 2026-05-27
**Scope:** Ch 1–6 (implemented) deep-pass on math correctness, code-prose parity, bibliography, cross-references, infrastructure, format compliance
**Supersedes:** `audits/2026-05-27_repo_content_quality_audit.md` (10-finding draft; reconciliation in §A)
**Artifact policy:** findings-only; remediation tracked via Track-B GH issues + Track-A inline fixes
**Format:** F-numbered findings, Track A/B/C, bracketed status (per `audits/2026-05-25_standards_vs_post_transformers.md` and memory `project_ssm_foundations_audit_format.md`)

## Audit Context

This audit deepens the untracked 2026-05-27 draft (`audits/2026-05-27_repo_content_quality_audit.md`) by leveraging three resources the draft did not access:

1. **Local canonical math PDFs** under `~/Claude/research-kb/fixtures/library_books/`: Hairer-Lubich-Wanner *Geometric Numerical Integration*, Trefethen-Bau *Numerical Linear Algebra*, Arnold *Mathematical Methods of Classical Mechanics* + *ODE*, Strogatz, Wiggins, Sauer, Goldstein. Used for math-correctness verification.
2. **Local arXiv PDFs** under `~/Claude/lever_of_archimedes/`, including **2603.15569** (Mamba-3 paper) — used to verify Ch 4/5/6 Mamba-3 claims directly rather than rely on summary memory.
3. **Research-KB MCP** (2,200 sources, 311K concepts, 745K relationships) — used to surface the canonical literature for each candidate finding and to extend the audit into negative-space (concepts central to a chapter's theme but absent from its prose).

The 2026-05-27 draft itself is solid: 10 well-evidenced findings with command-output appendix and concrete remediation steps. The verdict on it is *kept and extended*, not *replaced* — its A1–A10 are reconciled into F1–F11 in §A. Additional F-findings (F12–F28) come from the deeper resources above.

## Core Thesis: Four Compounding Debts

1. **Citation discipline lags claim ambition.** The book makes specific technical claims about Mamba-3 (13 uncited mentions, paper not in `bibliography.bib`), HiPPO-LegS condition-number bounds (no `<Cite>`, possibly overstated), Gauss-Legendre symplecticness (no direct `<Cite>` to Hairer-Lubich-Wanner Vol II VI.1 or to Sanz-Serna/Lasagni/Suris), and RK order-condition counts (numbers don't match the chapter's own Theorem 5.1). Three layers all fail at once: claims, citations, and source-checked numbers.

2. **Cross-reference machinery is declared but dead.** 14 theorem/definition IDs declared in Ch 1–6 (matching the docs' policy), 15 `<Figure>` blocks lacking `id=` (against the docs' policy), and **zero `<XRef>` usages anywhere** (so all 14 declared IDs are orphans). The `check-xref-labels` script explicitly tolerates missing figure IDs. The infrastructure is half-built; readers cannot navigate via stable references.

3. **Math-correctness drift in chapters the C1 pilot anchors on.** Ch 5 §5.2 line 96 contradicts the chapter's own Theorem 5.1; Ch 5 Exercise 5.1 claims to verify "the four order-conditions of order 4" but verifies conditions of orders 1, 2, 3 instead; Ch 1 §1.2 line 111 dismisses Jordan blocks "essentially always" while Ch 1 §1.3 line 138 uses a Jordan-block scenario (critical damping) as a canonical example. Ch 3 §3.3 makes a `κ` bound claim stronger than what the literature supports. The C1 pilot anchors on Ch 1–3 + Ch 6 (per CLAUDE.md). Two of those four chapters carry pilot-blocking correctness drift.

4. **Companion-code correctness and prose-parity drift.** Track B agent identified a **critical math bug** in `companions/ch04/julia/discretization_atlas.jl:99`: the Julia exp-trapezoidal implementation sets `M[n+1, n+2] = T(1.0)` instead of `dt`, producing a `B1` that is off by a factor of `dt`. The Julia exp-trap thus behaves as a first-order (not second-order) scheme. The Ch 4 Julia `runtests.jl` has **zero coverage** for `discretize_exp_trap`, so the bug is invisible to the gate. Additionally, three code/prose mismatches: Ch 4 §4.6 figure caption says spring constant $k=2$ but the script uses $k=4$; Ch 6 `stiff_demo.py` docstring says $\mu=30$ but the code uses $\mu=10$; Ex 6.3 solution cites a drift rate at $\Delta=0.05$ that the script never actually computes (it runs at $\Delta=0.3$). The chapter prose's claim that "both companions implement the augmented form" is now false (until the Julia bug is fixed).

Positive baseline (unchanged from draft):
- `make check` passes: 17 chapters validated, 16 bibkeys + 25 citations resolve, 14 theorem IDs declared, `docs/STATUS.md` 1.1 days old (≤14d freshness).
- `npm run build` passes; `npm audit --audit-level=moderate` reports 0 vulnerabilities.
- Python companions compile under `python3`; Julia companion tests pass for Ch 5 + Ch 6 (Ch 4 requires manual `Pkg.instantiate()`, now done).
- 16 entries in `bibliography.bib` and `src/data/references.json` are in sync.

## All-Findings Priority Table

| F# | Severity | Area | Finding (short) | Track | Lens | Status | A→F |
|---|---|---|---|---|---|---|---|
| F0  | IMPORTANT | Process | Audit-format drift: 2026-05-26 + 2026-05-27 drafts use A-numbers/severity-strings; canonical 2026-05-25 + memory codify F-numbers + Track A/B/C + bracketed status | A | Maint | `[open]` | new |
| F1  | CRITICAL  | Truthfulness | README chapter status stale (says Ch 1–3 active; reality Ch 1–6 `implemented`) | A | Maint | `[open]` | A1 |
| F2  | CRITICAL  | Correctness | Ch 5 §5.2 line 96 "5 conditions for order 4" — wrong number, contradicts the chapter's own Theorem 5.1 (one per Butcher tree of order ≤ p ⇒ cumulative 1,2,4,8,17,37) | A | Pilot | `[open]` | A2 (split) |
| F3  | CRITICAL  | Correctness | Ch 5 Exercise 5.1 claims to verify "the four order-conditions of order 4" but lists conditions of orders 1, 2, 3 (∑b=1, ∑b·c=1/2, ∑b·c²=1/3); zero order-4 conditions verified | A | Pilot | `[open]` | A2 (split) |
| F4  | CRITICAL  | XRef machinery | Figure IDs missing (15 `<Figure>`, 0 with `id=`) AND zero `<XRef>` usages in Ch 1–6 (14 declared IDs all orphans); `check-xref-labels.mjs:46-47` tolerates missing IDs | B | Pilot | `[open]` | A3 + X1 merged |
| F5  | IMPORTANT | Policy   | Notebook policy undocumented (zero `.ipynb`; post_transformers had 42 weekly notebooks; companion scripts are the implicit replacement but never stated) | C | Maint | `[open]` | A4 |
| F6  | CRITICAL  | Citations | Mamba-3 paper (arXiv:2603.15569, ICLR 2026) missing from `bibliography.bib` AND **13 uncited Mamba-3 mentions across Ch 2/3/4/6** | A | Pilot | `[open]` | A5 (extended) |
| F7  | IMPORTANT | Reproducibility | Ch 4 Julia test fails on fresh env until `Pkg.instantiate()` is run; default `make companion-julia-tests` excludes Ch 4 | B | Pilot | `[open]` | A6 |
| F8  | IMPORTANT | Docs drift | Scaffold version docs say v4.2.0; actual `package-lock.json` resolves v4.5.1. Docs use bare `python`; only `python3` available in this shell | A | Maint | `[open]` | A7 |
| F9  | MINOR     | Asset hygiene | Two committed figure files unused: `public/figures/ch01/matrix_exponential_convergence.png`, `public/figures/ch06/stiff_blowup.png` | A | Maint | `[open]` | A8 |
| F10 | MINOR     | Deps | Astro 6.3.7→6.3.8 patch available; KaTeX 0.16.47→0.17.0 minor available | C | Maint | `[open]` | A9 |
| F11 | MINOR     | Docs | CLAUDE.md:22 says exercises live in separate `chXX/exercises.mdx`; reality: embedded in chapter MDX | A | Maint | `[open]` | A10 |
| F12 | IMPORTANT | Correctness | Ch 1 §1.2 line 111 + Ch 3 §3.1 line 71–72 dismiss Jordan blocks as "essentially always invalid" / "essentially always valid diagonalizable" — but Ch 1 §1.3 line 138 uses critical-damped oscillator (a Jordan-block case) as canonical example. Internal inconsistency. Also fails for structured/learned A (HiPPO-LegS, S4) | A | Pilot | `[open]` | new |
| F13 | MINOR     | Correctness | Ch 2 §2.5 Theorem 2.3 BIBO statement (line 185–189): "poles in open LHP" — pole-zero-cancellation caveat appears as aside (line 189), creating pedagogical tension between theorem and footnote. Statement is mathematically correct | C | Maint | `[open]` | new |
| F14 | IMPORTANT | Correctness | Ch 3 §3.3 line 116–117 claims "HiPPO-LegS matrix keeps its condition number bounded as N grows" — no `<Cite>`. Research-kb shows literature says κ depends sub-quadratically on n (i.e., grows sub-quadratically, NOT bounded). Claim is likely **overstated**, not just uncited | A | Pilot | `[open]` | new |
| F15 | MINOR     | Correctness | Ch 4 §4.3 line 114 ZOH O(h²) per-step error: assumes u ∈ C¹ but regularity hypothesis not stated | C | Maint | `[open]` | new |
| F16 | IMPORTANT | Correctness | Ch 6 §6.5 line 135 "the *unique* s-stage RK method of order 2s" — missing "implicit" qualifier (and ideally "collocation" per Leimkuhler-Reich p.171). Section header says "IRK"; prose drops the qualifier | A | Pilot | `[open]` | new |
| F17 | MINOR     | Correctness | Ch 6 §6.4 line 103 Hamiltonian energy conservation: proof silently assumes H ∈ C¹; not stated | C | Maint | `[open]` | new |
| F18 | IMPORTANT | Citations | Ch 6 §6.5 line 135 Gauss-Legendre symplecticness asserted as fact; no direct `<Cite>` to Hairer-Lubich-Wanner Vol II §VI.1.2 or to Sanz-Serna/Lasagni/Suris originals (§6.4 line 93 cites HLW for general theory but not for symplecticness specifically) | A | Pilot | `[open]` | new |
| F19 | IMPORTANT | Bibliography | `gu2024mamba` is `@article{... journal="Conference on Language Modeling (COLM)"}`; COLM is a conference — should be `@inproceedings{... booktitle=...}` | A | Maint | `[open]` | new |
| F20 | IMPORTANT | Bibliography | `dao2024mamba2` same shape (`@article{... journal="ICML"}`) — should be `@inproceedings` | A | Maint | `[open]` | new |
| F21 | MINOR     | Bibliography | `anonymous2025lyapunov` has `Author=Anonymous`, OpenReview-only URL, no DOI; metadata will rot if paper moves venue | C | Maint | `[open]` | new |
| F22 | MINOR     | Process | Zero `confirmed_review_<date>` markers in `bibliography.bib`; review trail convention not adopted | C | Maint | `[open]` | new |
| F23 | INFO      | Bibliography | 4 entries cited nowhere in Ch 1–6 (`blelloch1990prefix`, `gu2020hippo`, `gu2022s4d`, `gu2024mamba`) — likely intentional forward refs to Ch 7+ stubs | C | Maint | `[open]` | new |
| F24 | IMPORTANT | Bibliography | Cross-repo diff against `post_transformers/references/dossier/*.bib`: ~259 canonical references present in predecessor but absent here. Most are Ch 7+ forward refs (acceptable to defer), but core foundations refs are missing (e.g., Khalil *Nonlinear Systems*, Chen 2018 Neural ODE) | C | Maint | `[open]` | new |
| F25 | MODERATE  | Structure | `ssm-foundations` uses flat `bibliography.bib`; post_transformers uses 7 topical dossier bibs (dynamical_systems_theory, ssm_lti, ssm_selective, etc.). Flat file will balloon as Ch 7–17 author; recommend dossier migration before Ch 10 | C | Maint | `[open]` | new |
| F26 | IMPORTANT | Code quality | JAX companions have no assertion-based test suite (no pytest, no `@test` analogue); pedagogical numerical claims in JAX are unverified. Julia has `runtests.jl` with `@test`; asymmetry | B | Maint | `[open]` | new (Track B agent) |
| F27 | IMPORTANT | Code quality | Torch parity gap: `companions/ch{01..06}/torch/` are all empty advertised dirs. Either populate or remove | B | Maint | `[open]` | new (Track B agent) |
| F28 | MINOR     | Code quality | Julia + JAX docstring coverage gaps: `ch06/julia/symplectic_methods.jl` 0%, `ch06/julia/implicit_methods.jl` 20%, `ch05/jax/order_verification.py` 27%, `ch06/jax/symplectic_demo.py` 50% (vs Ch 1–3 JAX = 100%) | A | Maint | `[open]` | new (Track B agent) |
| F29 | **CRITICAL** | Correctness (code) | **Julia exp-trap math bug**: `companions/ch04/julia/discretization_atlas.jl:99` sets `M[n+1, n+2] = T(1.0)` instead of `dt`; produces `B1` off by factor of `dt`; Julia exp-trap is first-order not second-order. Invisible to gate (no test). Falsifies the "both companions implement the augmented form" claim at `ch04-discretization.mdx:178` | A | Pilot | `[open]` | new (Track B agent) |
| F30 | IMPORTANT | Code-prose | `companions/ch04/jax/exp_trapezoidal.py:78` docstring shows `M[n,n+1]=1` in LaTeX but code at line 113 sets `dt`; docstring contradicts implementation | A | Maint | `[open]` | new (Track B agent) |
| F31 | IMPORTANT | Code-prose | `ch04-discretization.mdx:195` caption says $\ddot q + 0.5\dot q + 2q = \sin(2t)$ ($k=2$); `discretization_comparison.py:_K_STIFF=4.0` ($k=4$). Mismatch | A | Maint | `[open]` | new (Track B agent) |
| F32 | IMPORTANT | Code-prose | `ch06/jax/stiff_demo.py:13` module docstring says $\mu=30$; code at line 47 uses `_MU=10.0`. Mismatch | A | Maint | `[open]` | new (Track B agent) |
| F33 | IMPORTANT | Tests | `companions/ch04/julia/runtests.jl` has zero coverage for `discretize_exp_trap` (only ZOH + bilinear). This is why F29 was undetected | A | Pilot | `[open]` | new (Track B agent) |
| F34 | MODERATE | Code-prose | `ch06-implicit-and-symplectic.mdx:198` Ex 6.3 solution: "drift rate at $\Delta=0.05$ is ~$1.4 \times 10^{-8}$ per period" — but `symplectic_demo.py main()` only runs at $\Delta=0.3$; claimed number unreproducible from script | A | Maint | `[open]` | new (Track B agent) |
| F35 | MINOR    | Static analysis | `companions/ch02/jax/lyapunov_qr.py:46` has `import sys` after third-party imports — violates ruff `I001` (configured `select=["I"]` in pyproject); `# noqa: E402` suppresses only E402, not I001 | A | Maint | `[open]` | new (Track B agent) |
| F36 | MINOR    | Docs | `Makefile:27` help text says `companion-julia-tests` covers Ch 4/5/6; loop at line 79 only covers Ch 5/6 (Ch 4 deliberately excluded due to instantiate requirement) | A | Maint | `[open]` | new (Track B agent) |
| F37 | MINOR    | Deps | `companions/_shared/pyproject.toml` uses minimum-only pins (`jax>=0.4.30` etc.); no upper bounds; JAX has history of minor-version breaks | C | Maint | `[open]` | new (Track B agent) |

**Pilot-imminent gate** (≤14 days, anchored on C1 pilot Ch 1–3 + Ch 6): F2, F3, F4, F6, F12, F14, F16, F18, **F29, F33** promote to Track A (10 findings). F29 is the highest-impact: a real math bug in pilot-anchored Ch 4 companion code.

## Per-Finding Detail

### F0 — Audit-Format Drift (meta-finding)

**Evidence**

- Inaugural `audits/2026-05-25_standards_vs_post_transformers.md` establishes: F-numbered findings, Track A/B/C taxonomy, bracketed `[open]/[tracked]/[fixed]/[pilot-blocked]/[withdrawn]` status labels, pilot-imminent gate.
- Memory `project_ssm_foundations_audit_format.md` codifies this format.
- `audits/2026-05-26_first-deploy.md` deviates: A-numbered, "Fix next/Decide soon" priority strings, no Track classification, no bracketed status.
- `audits/2026-05-27_repo_content_quality_audit.md` copies the 2026-05-26 deviation.

**Impact**

Two consecutive audits drifted from canonical format. Without alignment, `grep [fixed]` and similar audit-trail tools degrade; future readers can't tell which audits are authoritative.

**Recommended fix**

Adopt the canonical format going forward (this audit does). Optionally retro-renumber the 2026-05-26 + 2026-05-27 drafts; not urgent if a reconciliation appendix exists.

### F2 — Ch 5 §5.2 RK Order-Condition Count Wrong

**Evidence**

- `src/content/chapters/ch05-stability-regions.mdx:96`: "5 conditions for order 4, 17 for order 5, 37 for order 6" with `<Cite key="hairer1993ordinary" />`.
- The chapter's own Theorem 5.1 (line 98–100): "of order $p$ ... iff a specific finite set of polynomial conditions on $(A,b,c)$ holds (**one per Butcher tree of order ≤ p**)."
- Butcher-tree cumulative counts (Hairer-Nørsett-Wanner Vol I §II.2; RootedTrees.jl tutorial; Leimkuhler-Reich p.171 confirms framework): 1, 2, 4, 8, 17, 37 (orders 1–6).
- Exact-order counts: 1, 1, 2, 4, 9, 20.
- Neither 4 nor 8 equals 5. Order 4 should be 8 (cumulative) or 4 (exact-order). The "5" is wrong by either reading.

**Impact**

A reader using the chapter as a study guide gets a wrong canonical count. The chapter explicitly tells readers Butcher's tree theory is the right framework via Theorem 5.1, then immediately contradicts it.

**Recommended fix**

Replace "5 conditions for order 4, 17 for order 5, 37 for order 6" with "the cumulative number of conditions for order $p$ is 1, 2, 4, 8, 17, 37 for $p = 1, 2, 3, 4, 5, 6$" (consistent with line 95's "Order 4: four more conditions"). Optionally cite RootedTrees.jl or HNW Vol I Theorem II.2.13 directly.

### F3 — Ch 5 Exercise 5.1 Mislabel

**Evidence**

- `ch05-stability-regions.mdx:196-198`: "Verify that classical RK4 satisfies the four order-conditions of order 4 — at least the simpler ones: $\sum b_i = 1$, $\sum b_i c_i = 1/2$, $\sum b_i c_i^2 = 1/3$."
- The three listed conditions are the conditions for orders 1, 2, 3 respectively (one tree each at order 1 and 2; one of two trees at order 3). **None** is an order-4 condition.
- The actual four order-4 Butcher-tree conditions are: $\sum_i b_i c_i^3 = 1/4$, $\sum_{i,j} b_i c_i a_{ij} c_j = 1/8$, $\sum_{i,j} b_i a_{ij} c_j^2 = 1/12$, $\sum_{i,j,k} b_i a_{ij} a_{jk} c_k = 1/24$.

**Impact**

Reader who completes Exercise 5.1 has not verified RK4 order-4. They've verified order ≤ 3. The exercise as written gives false confidence in the method's order-4 property.

**Recommended fix**

Rewrite Exercise 5.1 either (a) as "verify a subset of RK4 conditions through order 3" (true statement matching what's listed), OR (b) add the four genuine order-4 conditions to the exercise. Option (b) is pedagogically stronger.

### F4 — Cross-Reference Machinery Declared but Dead

**Evidence**

- `grep -c '<Figure' src/content/chapters/*.mdx` → 15 figures; `grep -c '<Figure.*id=' src/content/chapters/*.mdx` → 0.
- `grep 'id="ch' src/content/chapters/*.mdx | sort -u` → 14 declared theorem/definition IDs: `ch01:def:matexp`, `ch02:def:lyap`, `ch02:def:stab-region`, `ch02:thm:bibo-tf`, `ch02:thm:lyap-eig`, `ch03:thm:eig-decomp`, `ch03:thm:svd`, `ch04:bilinear-stability`, `ch04:lax-equivalence`, `ch04:zoh-stability`, `ch05:dahlquist-barrier`, `ch05:order-conditions`, `ch06:be-a-stable`, `ch06:symplectic-modified-hamiltonian`.
- `grep -rh '<XRef' src/content/chapters/*.mdx | wc -l` → **0**.
- `CLAUDE.md:24`: "Cross-references: `id='...'` on `<Theorem>` / `<Figure>`".
- `STYLE.md` §4: "every `<Theorem>` and `<Figure>` block should use `id='ch##:<type>:<slug>'`".
- `scripts/check-xref-labels.mjs:46-47`: `if (!hasId) continue; // missing id is not required (yet)`.

**Impact**

The cross-reference infrastructure is declared in docs, partially built (theorem IDs only, no figure IDs, no XRef consumers), and never validated. The book cannot internally link results across chapters. CI tolerates the gap.

**Recommended fix**

Decide policy:
- **Option A (full enforcement)**: add `id=` to all 15 figures; introduce `<XRef>` in cross-chapter references (Ch 2 should `<XRef>` Ch 1 §1.2 matrix-exponential; Ch 5 should `<XRef>` Ch 4 Lax equivalence; etc.); flip `check-xref-labels.mjs` to fail on missing figure IDs.
- **Option B (retract docs)**: update `STYLE.md` and `CLAUDE.md` to say only theorems/definitions get IDs; remove figure-ID language.

Recommended: Option A. Scaffold's `build-labels` supports figures (per `scripts/check-xref-labels.mjs`); enforcement is the missing piece.

### F6 — Mamba-3 Citation Gap

**Evidence**

- `bibliography.bib` has `gu2024mamba` (Mamba-1) and `dao2024mamba2` (Mamba-2). **No Mamba-3 entry**.
- `lever_of_archimedes/2603.15569.pdf` (locally available): Mamba-3 paper, ICLR 2026 oral. Title: "Mamba-3: Improved Sequence Modeling using State Space Principles" (Lahoti et al. per Track F agent's cross-repo diff; entry should be `lahoti2026mamba3` or similar).
- Track C agent grep: **13 uncited Mamba-3 mentions** across `ch02-stability-theory.mdx:169`, `ch03-linear-algebra.mdx:118`, `ch04-discretization.mdx:33, 52, 176, 197, 203`, `ch06-implicit-and-symplectic.mdx:44, 46, 74, 85` plus 2 more in Ch 5.

**Direct verification from local PDF (Track F agent)**:
- "Exponential-trapezoidal discretization" claim (Ch 4): ✓ ACCURATE — paper Proposition 1 pp. 5–6.
- "A-stability for exp-trapezoidal" (Ch 2): UNCITED-BUT-CORRECT-WITH-CAVEAT — paper Section 5.4 frames as "second-order" without explicit A-stability proof; A-stability relies on external exponential-integrator theory (Hochbruck 2010).
- "Stiffness motivation" (Ch 4): UNCITED-BUT-CORRECT — paper acknowledges informally; ssm-foundations makes it formal.
- "Complex-state design" (Ch 3): ✓ ACCURATE — paper Proposition 2.
- "MIMO advantage over backward-Euler" (Ch 6): ✓ ACCURATE — paper §3.3 + Table 6.
- "Mamba-3 does not use ETDRK4" (Ch 6): UNCITED-BUT-REASONABLE-INFERENCE — paper doesn't discuss; this is author commentary.

**Recommended fix**

Add Mamba-3 BibTeX entry to `bibliography.bib`:
```bibtex
@inproceedings{lahoti2026mamba3,
  author    = {Lahoti, Karan and ...},
  title     = {Mamba-3: Improved Sequence Modeling using State Space Principles},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2026},
  note      = {arXiv:2603.15569}
}
```
Add `<Cite key="lahoti2026mamba3" />` near each of the 13 uncited claim sites. For claims that are "uncited-but-correct-with-caveat" (A-stability), additionally cite Hochbruck-Ostermann *Exponential Integrators* (2010) or equivalent external source.

### F12 — Jordan-Block Dismissal Internal Inconsistency

**Evidence**

- `ch01-linear-odes.mdx:111`: "In practice almost every $\statemat$ arising from random initialization or from physical systems is diagonalizable, and the polynomial-factor case shows up only at codimension-one boundaries in parameter space."
- `ch01-linear-odes.mdx:138`: "**Critically damped** ($c^2 = 4k$): a repeated negative eigenvalue $\lambda = -c/2$. The matrix $\statemat$ is in this case *not* diagonalizable, and the solution contains a $t \cdot e^{\lambda t}$ term."
- `ch03-linear-algebra.mdx:71-72`: "almost every $\statemat$ arising from random initialization or from training is diagonalizable ... **In practice the assumption '$\statemat$ is diagonalizable' is essentially always valid; the Jordan form is the worst-case fallback.**"
- `ch02-stability-theory.mdx:64`: Theorem 2.1(1) explicitly carves out the defective-imaginary case ("its algebraic multiplicity equals its geometric multiplicity (no defective Jordan blocks)").

The chapter dismisses Jordan blocks as essentially nonexistent, then immediately uses one (critical damping in §1.3 line 138) as a canonical example, then Ch 2's Lyapunov theorem requires the careful Jordan caveat. The story is internally inconsistent.

Additionally, the dismissal is wrong for the SSM application context:
- HiPPO-LegS (Ch 3 §3.3) is a structured matrix designed for specific spectral properties; its eigenstructure is not generic.
- S4's diagonal-plus-low-rank parametrization (cited in Ch 3 §3.5) deliberately produces structured A.
- Critically damped boundary IS a designed-for physical regime, not a measure-zero anomaly.

**Recommended fix**

Soften line 111 to: "For generic dense random $\statemat$, non-diagonalizability is measure-zero; however, structured/learned $\statemat$ (HiPPO-LegS, S4 DPLR, critical damping) may exhibit Jordan structure by design. Ch 2's Lyapunov theorem and Ch 1 §1.3 critical-damping example both rely on the Jordan form." Similarly soften Ch 3 §3.1 lines 71-72.

### F14 — HiPPO-LegS κ-Bounded Claim May Be Overstated

**Evidence**

- `ch03-linear-algebra.mdx:116-117`: "**HiPPO matrix construction.** The HiPPO-LegS matrix has structure that keeps its condition number bounded as $N$ grows — this is *why* HiPPO is the standard initialization."
- No `<Cite>` immediately near this claim.
- Figure 3.1 caption (`ch03-linear-algebra.mdx:120`): "HiPPO-LegS matrix (κ stays bounded as N grows — its design feature)".
- Research-kb search "HiPPO LegS condition number bounded" returned arXiv:2310.01698 Theorem 5: "the condition number κ($\tilde V_H$) should depend **linearly on ‖E‖⁻¹** and depend **sub-quadratically on n**". Sub-quadratic growth is unbounded growth, not bounded.
- Other research-kb chunks discuss HiPPO-LegS as having structure that admits *stable* diagonalization (DPLR form) — different claim from "κ bounded".

**Impact**

The chapter asserts a stronger property than the literature supports. The figure caption repeats the claim. A reader who tries to verify this against the original HiPPO paper (Gu et al. 2020, NeurIPS) will need to dig into the proofs to see what's actually claimed.

**Recommended fix**

(a) Add direct `<Cite>` to the HiPPO paper and/or the S4 paper near this claim. (b) Soften the claim from "bounded" to "grows slowly with N (sub-quadratically; see [HiPPO paper Lemma/Theorem]); typical $N \le 256$ remains tractable." (c) Update Figure 3.1 caption to match.

### F16 — Ch 6 Gauss-Legendre "Unique" Claim Under-Qualified

**Evidence**

- `ch06-implicit-and-symplectic.mdx:135`: "**Gauss–Legendre IRK** (order $2s$, A-stable, symplectic). The $s$-stage Gauss–Legendre Runge–Kutta method has nodes at the Gauss–Legendre quadrature points and is the *unique* $s$-stage RK method of order $2s$."
- Section header uses "IRK" but the prose drops the "implicit" qualifier.
- Canonical source (Leimkuhler-Reich *Simulating Hamiltonian Dynamics* p.171, via research-kb): "Gauss-Legrendre methods with $s \ge 1$ stages are of order $p = 2s$. This is the optimal order obtainable for a given number of stages **among all possible symplectic collocation Runge-Kutta methods**."
- Canonical source (Quarteroni-Sacco-Saleri *Numerical Mathematics* p.534-535): "Gauss-Legendre RK methods ... attain the maximum possible order $2s$" — context is *implicit* RK methods.
- For $s \ge 2$ the claim is essentially vacuous-then-strengthened (explicit RK of $s$ stages caps at order $s$, not $2s$), but at $s=1$ uniqueness "among RK methods" without qualifier is wrong: forward Euler is also a 1-stage RK method (of order 1, not 2 — but the prose's "unique 1-stage RK of order 2" only makes sense among *implicit* methods).

**Recommended fix**

Replace "the *unique* $s$-stage RK method of order $2s$" with "the *unique* $s$-stage **implicit** RK method achieving order $2s$ (the maximum possible for $s$ stages; see Hairer-Lubich-Wanner Vol II §IV.5 or Leimkuhler-Reich §6.3.1)".

### F18 — Gauss-Legendre Symplecticness Cite Missing

**Evidence**

- `ch06-implicit-and-symplectic.mdx:135`: Gauss-Legendre asserted to be symplectic.
- §6.4 line 93 cites `hairer1996ordinary` for "full theory" of Hamiltonian systems — but this is the stiff-ODE volume (HNW Vol II "Solving Ordinary Differential Equations II: Stiff and Differential-Algebraic Problems"). The symplecticness result is in the *Geometric Numerical Integration* volume (Hairer-Lubich-Wanner Springer Series in Comp Math Book 31).
- Research-kb confirms the canonical sources for Gauss-Legendre symplecticness: Sanz-Serna (1988), Lasagni (1988), Suris (1989). Also Leimkuhler-Reich p.168 cites these originals + p.171 contains the symplecticness statement directly.
- `bibliography.bib` has `hairer1996ordinary` (Vol II stiff) but no entry for HLW Geometric Numerical Integration (the *correct* citation for symplecticness).

**Recommended fix**

Add a new bib entry `hairer2006geometric` for *Geometric Numerical Integration* (Springer 2006/2010, 2nd ed.). Add `<Cite key="hairer2006geometric" />` after the "symplectic" claim in line 135. Optionally also cite Sanz-Serna 1988 or Leimkuhler-Reich for the original symplecticness result.

### F19, F20 — BibTeX Type Errors on Mamba Entries

**Evidence (Track C agent)**

- `bibliography.bib:134-140` (`gu2024mamba`): `@article{... journal="Conference on Language Modeling (COLM)"}`. COLM is a conference.
- `bibliography.bib:142-149` (`dao2024mamba2`): `@article{... journal="International Conference on Machine Learning (ICML)"}`. ICML is a conference.

**Recommended fix**

Change `@article` → `@inproceedings` and `journal=` → `booktitle=` for both entries. After fix, regenerate `src/data/references.json` via `npm run build:bib`.

### F26 — JAX Test-Suite Absence

Julia has `companions/ch{05,06}/julia/runtests.jl` with `@test` assertions verifying pedagogical claims (RK order=4, energy band bounded, etc.). JAX has no equivalent. The Track B agent points out this is exactly why F29 (Julia exp-trap bug) survived — and a similar undetected error could exist in any JAX companion. Recommend adding `companions/ch{01..06}/jax/tests/test_*.py` with parametrized pytest assertions for order, stability, and conservation claims.

### F27 — Torch Parity Gap

`companions/ch{01..03}/torch/.gitkeep` (empty placeholders); `companions/ch{04..06}/torch/` don't exist at all. Asymmetry within Ch 1–6. No chapter MDX references torch companions by path. CLAUDE.md is silent on whether torch is deferred or expected. `companions/_shared/pyproject.toml:28-30` has `[torch]` optional extra.

Recommended: delete the three `.gitkeep` files, document deferral in `companions/README.md` and CLAUDE.md, retain the `[torch]` optional extra for future use.

### F28 — Docstring Coverage Gaps

Track B agent measured public-function docstring coverage:

| File | Public fns | With docstring | % |
|---|---|---|---|
| `ch01/jax/*.py` (3 files) | 11 | 11 | 100% |
| `ch02/jax/lyapunov_qr.py` | 4 | 4 | 100% |
| `ch02/jax/stability_regions.py` | 3 | 2 | 67% |
| `ch03/jax/*.py` (2 files) | 10 | 8 | 80% |
| `ch04/jax/*.py` (2 files) | 19 | 18 | 95% |
| `ch05/jax/stability_regions.py` | 14 | 6 | 43% |
| `ch05/jax/order_verification.py` | 11 | 3 | **27%** |
| `ch06/jax/stiff_demo.py` | 7 | 3 | 43% |
| `ch06/jax/symplectic_demo.py` | 12 | 6 | 50% |
| `ch06/julia/implicit_methods.jl` | 5 | 1 | 20% |
| `ch06/julia/symplectic_methods.jl` | 7 | 0 | **0%** |

Ch 1–4 files are uniformly well-documented (67–100%). Ch 5–6 files are systematically under-documented. `pyproject.toml:31` mypy strict and `select=["I"]` in ruff imply the standard; these files violate it.

### F29 — Julia Exp-Trap Implementation Bug (CRITICAL)

**Evidence**

`companions/ch04/julia/discretization_atlas.jl:99`:
```julia
M[n+1, n+2] = T(1.0)
```

This is the (N+1, N+2) entry in the 1-based augmented matrix `M = [[A·dt, B·dt, 0]; [0, 0, ·]; [0, 0, 0]]`. The Python counterpart at `companions/ch04/jax/exp_trapezoidal.py:113`:
```python
M.at[n, n+1].set(dt)
```
is correct: 0-based (N, N+1) entry set to `dt`.

The augmented-matrix-exponential identity (Exercise 4.4 solution in `ch04-discretization.mdx:259-267`) requires the off-diagonal block to contain `dt` (or equivalently, the augmented matrix to have a `dt` in the position that produces a `dt²` factor after exponentiation, which after dividing by `dt` gives the required `dt · φ₂(A·dt) · B`).

With `T(1.0)` instead of `dt`, the resulting `B1` matrix is off by a factor of `dt`. The exp-trap step `step_exp_trap` then applies a B1 that is `1/dt` times too small. For smooth forcing where $u_{k+1} - u_k = O(dt)$, the correction term becomes $O(1)$ instead of $O(dt)$ — degrading exp-trap from second-order to first-order accuracy.

**Why undetected**: `companions/ch04/julia/runtests.jl` covers `discretize_zoh` (2 tests), `discretize_bilinear` (1 test), and a generic ZOH convergence test (1 test). **Zero tests for `discretize_exp_trap`**.

**Impact**

This is pilot-blocking. The C1 pilot anchors on Ch 1–3 + Ch 6 but uses Ch 4 exp-trap as a building block (Mamba-3 = exp-trap; Ch 10 + pilot work depend on Ch 4). The chapter prose at `ch04-discretization.mdx:178` claims "Both companions implement the augmented form" — false until fixed.

**Recommended fix**

```julia
# companions/ch04/julia/discretization_atlas.jl:99
M[n+1, n+2] = dt   # was: T(1.0)
```

Also fix comments on lines 102-104 and add a convergence-order test to `companions/ch04/julia/runtests.jl` (F33):
```julia
@testset "Exp-trapezoidal achieves order 2" begin
    # compare slopes between exp_trap and ZOH on a forced linear problem
    ...
end
```

### F30 — JAX Exp-Trapezoidal Docstring Contradicts Implementation

`companions/ch04/jax/exp_trapezoidal.py:78` docstring shows the augmented matrix as
```latex
M = [[A, B, 0], [0, 0, 1], [0, 0, 0]]
```
(with `1` in position [n, n+1]). Code at line 113 sets `M.at[n, n+1].set(dt)` — the actual value is `dt`, not `1`. The docstring's LaTeX is wrong; the implementation is correct.

**Recommended fix**: Update the docstring LaTeX to show `dt` in position [n, n+1].

### F31, F32 — Code-Prose Constant Mismatches

**F31**: `ch04-discretization.mdx:195` figure caption: "the forced damped oscillator $\ddot q + 0.5 \dot q + 2 q = \sin(2t)$" — so $k=2$, $c=0.5$. But `companions/ch04/jax/discretization_comparison.py` has `_K_STIFF = 4.0` (and the caption's $k=2$ matches Ch 1's example, not the Ch 4 simulation). Either update the caption to match `k=4` or update the code constant.

**F32**: `companions/ch06/jax/stiff_demo.py:13` module docstring: "van der Pol oscillator at $\mu = 30$ (mildly stiff)". Code at line 47: `_MU: float = 10.0`. The chapter at `ch06-implicit-and-symplectic.mdx:262` says "van der Pol oscillator at $\mu = 30$ (mildly stiff)" — so the chapter agrees with the docstring, both contradict the code. Either update `_MU = 30.0` or update both the docstring and the chapter prose.

### F33 — Ch 4 Julia Missing Exp-Trap Test

`companions/ch04/julia/runtests.jl` has zero `@test` coverage for `discretize_exp_trap` (covers ZOH, bilinear, but not exp-trap). This is why F29 was undetected. Adding the missing test is a direct prerequisite for closing F29.

### F34 — Ex 6.3 Cites Unreproducible Drift Rate

`ch06-implicit-and-symplectic.mdx:198`: Ex 6.3 solution claims "the drift rate at $\stepsize = 0.05$ is roughly $1.4 \times 10^{-8}$ per period". But `companions/ch06/jax/symplectic_demo.py main()` runs only at $\Delta = 0.3$ (figure config). A reader running the script cannot verify the cited number. Either add a $\Delta=0.05$ run to `main()` (low cost) or change the exercise solution to reference $\Delta=0.3$ values.

### F35 — Ruff I001 Violation in `lyapunov_qr.py`

`companions/ch02/jax/lyapunov_qr.py:46` has `import sys` after `matplotlib`, `numpy`, `scipy`, and a `companions._shared` import. `companions/_shared/pyproject.toml:41` configures `ruff.lint.select = ["E","F","W","I","N","UP","B","SIM"]` — `I` is isort/import-order. The `# noqa: E402` comment on line 48 suppresses only E402, not I001. Move `import sys` to the stdlib block above third-party imports.

### F36 — Makefile Help Text Mismatch

`Makefile:27` help text describes `companion-julia-tests` as running Ch 4/5/6; the loop at line 79 only runs Ch 5/6 (Ch 4 deliberately excluded due to instantiation requirement). Fix the help text to say "Ch 5, Ch 6 (Ch 4 excluded; run manually after `Pkg.instantiate()`)".

### F37 — Dependency Pins Minimum-Only

`companions/_shared/pyproject.toml`: `jax>=0.4.30`, `jaxlib>=0.4.30`, `matplotlib>=3.9.0`, `numpy>=1.26.0`, `scipy>=1.14.0` — minimum-only pins. JAX has historically broken APIs across minor versions. Recommend adding tested upper bounds (e.g., `jax>=0.4.30,<0.6`) once the safe range is known.

## Asymmetries

- **Citation asymmetry**: claims-density >> citation-density. Mamba-3 (13 uncited mentions, no bib entry) is the worst case; HiPPO-LegS κ-bound (no cite, possibly overstated) is the second worst.
- **Cross-reference asymmetry**: docs say everything should have IDs; reality is theorems do (14), figures don't (0/15), XRef consumers don't exist (0).
- **Companion asymmetry**: JAX coverage (Ch 1–6) > Julia (Ch 4–6) > torch (none). JAX is most-covered but least-tested; Julia is least-covered but most-tested.
- **Pilot-relevance asymmetry**: Ch 1, 3, 5, 6 (4 of the 6 implemented chapters) all have pilot-imminent correctness findings (F2, F3, F12, F14, F16, F18); Ch 2, 4 only have minor findings. The C1 pilot anchors on Ch 1–3 + Ch 6 — three of those four chapters have findings promoted to Track A under pilot-imminent gate.

## Verification Appendix

All evidence in this audit was produced by the following commands, run from `/home/brandon_behring/Claude/ssm-foundations` on 2026-05-27.

```bash
# Baseline content gate
make check
# → all gates passed: 17 chapters, 16 bibkeys (25 citations / 12 keys), 14 theorem IDs, status 1.1d

# XRef + Figure ID inventory (F4)
grep -rh 'id="ch' src/content/chapters/*.mdx | sort -u | wc -l   # → 14
grep -rhE '<Figure[^>]*id=' src/content/chapters/*.mdx | wc -l   # → 0
grep -rh '<XRef' src/content/chapters/*.mdx | wc -l              # → 0
grep -rh '<Cite' src/content/chapters/*.mdx | wc -l              # → 25

# BibTeX type errors (F19, F20)
grep -A2 '^@article{gu2024mamba\|^@article{dao2024mamba2' bibliography.bib

# Mamba-3 mention inventory (F6)
grep -rn -iE 'mamba.?3|mamba 3' src/content/chapters/*.mdx       # → 13 hits across 4 chapters

# Bibliography review-trail markers (F22)
grep 'confirmed_review_' bibliography.bib                         # → 0 hits

# Julia tests (F7, F28)
make companion-julia-tests                                        # Ch 5+6 pass; Ch 4 separate
julia --project=companions/ch04/julia -e 'using Pkg; Pkg.instantiate()' && \
  julia --project=companions/ch04/julia companions/ch04/julia/runtests.jl   # Ch 4 needs instantiate

# JAX install (Track B agent in progress)
pip install --user -e companions/_shared                          # Exit 0
```

**Research-KB MCP queries** (Track E, supporting F2, F16, F18, F14):
- `research_kb_search("Runge-Kutta order conditions Butcher trees ...", domain="numerical_methods")` → 5 hits, all Hairer-Lubich-Wanner GNI Vol III §III.1 + §III.2 + §III.3 (Source ID `fdd95e81-6e4e-4804-aedc-2319ab54cff0`).
- `research_kb_search("Gauss-Legendre symplectic implicit ...", domain="numerical_methods")` → Leimkuhler-Reich p.171 (Source ID `b6029362-1b02-4133-b0cf-24ab3ade56dd`), Quarteroni-Sacco-Saleri p.534-535 (Source ID `8a8de81c-f312-4bc5-8512-771850bfc6be`).
- `research_kb_search("HiPPO LegS condition number bounded")` → arXiv:2310.01698 Theorem 5 (Source ID `86405a83-459f-4e20-a507-d6dbb62b82eb`): "sub-quadratic" growth, not bounded.

**Direct PDF read**: `/home/brandon_behring/Claude/lever_of_archimedes/2603.15569.pdf` (Mamba-3 paper) — Track F agent verified 6 specific Mamba-3 claims; see F6.

## §A — Reconciliation Appendix: A1–A10 → F-numbers

| Draft # | This audit's # | Status | Notes |
|---|---|---|---|
| A1 (README stale) | F1 | **confirmed** | Wording preserved; promoted to CRITICAL+Track A |
| A2 (Ch 5 RK counts) | **F2 + F3** (split) | **extended** | Probe split into count-error (F2) AND exercise-mislabel (F3, new) |
| A3 (figure IDs missing) | **F4** (merged with X1) | **extended** | Merged with zero-XRef-usage finding into "machinery half-built" |
| A4 (notebook policy) | F5 | **confirmed** | Wording preserved |
| A5 (Mamba-3 missing) | **F6** (extended) | **extended** | 13 uncited mentions added; direct PDF verification added |
| A6 (Ch 4 Julia instantiate) | F7 | **confirmed** | |
| A7 (docs drift v4.2.0/v4.5.1) | F8 | **confirmed** | |
| A8 (unused figure assets) | F9 | **confirmed** | |
| A9 (Astro/KaTeX outdated) | F10 | **confirmed** | |
| A10 (CLAUDE.md exercise layout) | F11 | **confirmed** | |
| (new) | F0 | meta | Format compliance drift |
| (new) | F12 | extends Ch 1 + Ch 3 dismissal of Jordan blocks |
| (new) | F13 | Ch 2 BIBO pedagogical tension (minor) |
| (new) | F14 | Ch 3 HiPPO κ claim possibly overstated |
| (new) | F15 | Ch 4 ZOH regularity assumption (minor) |
| (new) | F16 | Ch 6 Gauss-Legendre "unique" qualifier missing |
| (new) | F17 | Ch 6 Hamiltonian C¹ assumption (minor) |
| (new) | F18 | Ch 6 Gauss-Legendre symplecticness cite missing |
| (new) | F19, F20 | Mamba BibTeX type errors |
| (new) | F21 | Anonymous lyapunov metadata speculative |
| (new) | F22 | No review-trail markers |
| (new) | F23 | Cited-nowhere entries (info-only) |
| (new) | F24 | 259+ missing refs vs post_transformers |
| (new) | F25 | Dossier-organization gap |
| (new) | F26 | JAX test-suite absence (Track B agent) |
| (new) | F27 | Torch parity gap (Track B agent) |
| (new) | F28 | Docstring coverage gaps with detailed per-file metrics (Track B agent) |
| (new) | **F29** | **Julia exp-trap math bug — CRITICAL, pilot-blocking** (Track B agent) |
| (new) | F30 | JAX exp-trap docstring contradicts impl (Track B agent) |
| (new) | F31 | Ch 4 figure caption k=2 vs code k=4 (Track B agent) |
| (new) | F32 | Ch 6 stiff_demo μ=30 docstring vs μ=10 code (Track B agent) |
| (new) | F33 | Ch 4 Julia missing exp-trap @test (Track B agent) |
| (new) | F34 | Ex 6.3 unreproducible drift-rate claim (Track B agent) |
| (new) | F35 | Ruff I001 violation in lyapunov_qr.py (Track B agent) |
| (new) | F36 | Makefile help text mismatch (Track B agent) |
| (new) | F37 | Deps minimum-only pins (Track B agent) |

**Net additions**: 27 new F-findings beyond A1–A10 (plus F0 meta). **Net structural changes**: 1 split (A2→F2+F3), 1 merge (A3+X1→F4), 1 extension (A5→F6 with direct PDF verification).

**Most consequential single finding not in the draft**: F29 (Julia exp-trap implementation bug). This is a real numerical error in production code, invisible to the existing CI gate, falsifying a chapter-prose claim of cross-companion parity, and located in pilot-anchored Ch 4. The existing audit didn't find it because it only ran `py_compile` (syntax) and `make companion-julia-tests` (which excludes Ch 4 and has no exp-trap test even for Ch 4). The deeper audit found it via direct source-comparison between the Python and Julia implementations of the same augmented-matrix-exponential identity.

## Overall Assessment

The repo's infrastructure remains healthy and CI gates remain green. The shift from the 2026-05-27 draft is that the deeper-research pass found:

- **Citation discipline gap is larger than A5 indicated** — Mamba-3 alone has 13 uncited mentions of a paper not even in `bibliography.bib`. Add to that HiPPO κ-bound (F14), Gauss-Legendre symplecticness (F18), Jordan-block dismissal pattern (F12), and the picture is systemic, not localized.
- **Math correctness drift hits pilot-anchored chapters** — Ch 5 (F2 + F3), Ch 6 (F16 + F18), Ch 3 (F14), Ch 1 (F12). The C1 pilot anchors on Ch 1–3 + Ch 6 (per CLAUDE.md §"Pilot integration policy"); three of those four chapters now have promoted-to-Track-A findings under the pilot-imminent gate.
- **Cross-reference machinery is dead** (F4) — beyond A3's figure-ID note, the deeper finding is that the entire XRef infrastructure has zero consumers. This is fixable in one pass but should be done before Ch 7+ are authored, since cross-chapter references multiply quickly.

Recommended remediation order:

1. **F29 first** (truly critical, code-correctness bug, blocks pilot, two-line fix): apply the diff at `companions/ch04/julia/discretization_atlas.jl:99` (change `T(1.0)` → `dt`) and add convergence-order `@test` for exp-trap to `runtests.jl` (closes F33 simultaneously).
2. **Track A pilot-imminent (≤14d)**: F2, F3, F4, F6, F12, F14, F16, F18, F33 (eight findings that block or threaten C1 pilot anchoring).
3. **Track A code-prose parity**: F30, F31, F32, F34 — small same-session fixes (docstrings, captions, code constants).
4. **Track A maintenance**: F0, F1, F8, F9, F11, F19, F20, F28, F35, F36 (quick wins, same-session).
5. **Track B (GH-tracked)**: F4 (XRef enforcement), F7 (Ch 4 Julia gate), F26, F27 (code quality systemic).
6. **Track C (deferred)**: F5, F10, F13, F15, F17, F21, F22, F23, F24, F25, F37.

The deeper resources (local canonical PDFs + research-kb MCP + post_transformers/references dossiers + direct Julia source inspection via the code-reviewer agent) significantly strengthened the audit's grounding. Future audits should leverage these by default. **The single most important takeaway**: the existing test gate is *insufficient* — it found nothing wrong with the Ch 4 Julia companion, yet a real numerical bug (F29) was present. The JAX-test-suite absence (F26) and Ch 4 exp-trap-test absence (F33) are structural causes; adding parametrized convergence-order tests across both languages is the cheapest path to durable correctness assurance.
