# Chapter research brief — Ch 12: Delta-rule lineage

> Filled 2026-06-10 from the campaign-plan recon (predecessor `week12/` inventory,
> forward-promise grep, `labels.json`, `bibliography.bib`). Drives the step-0
> `/exploring-options` round via §6.

- **Chapter / slug:** ch 12 — `ch12-delta-rule-lineage.mdx`
- **Part / status target:** `beyond-ssm` → `implemented`
- **One-line scope:** State updates as *online learning* — the delta rule is one gradient step
  on an associative-recall loss; DeltaNet (explicit Euler) and Longhorn (implicit step) are the
  textbook discretization pair from Ch 4–6 applied to that ODE; gating is forgetting; chunkwise
  is the production form.
- **Pilot tie-in:** B (two-timescale) — the online-learning ODE underpins B's
  implicit-vs-explicit gating analysis, and the unified attention↔SSM two-limit view seeds B's
  singular-perturbation framing (attention = boundary layer, SSM = slow manifold). No C1 need.

## 1. Forward-promises to redeem

| Source (file:line) | Promise made | How this chapter honours it |
|---|---|---|
| `ch11:48-50` | "linear-attention + gating vocabulary built here is the prerequisite for Chapter 12's delta-rule lineage (the *overwrite* answer to §11.6's capacity limit)" | Open §12.1 from the §11.6 capacity wall; gating section builds on GLA decay masks |
| `ch11:311` | capacity gap closes via "a smarter *write* rule, which is the subject of Chapter 12" | The delta-rule write IS the chapter's object; revisit the MQAR capacity story with the delta-rule write |
| `ch11:322-333` | wrote the rank-one update $S_t=(I-\beta_t\phi(k_t)\phi(k_t)^\top)S_{t-1}+\beta_t\phi(k_t)v_t^\top$ and committed: "**Chapter 12** (the online-learning reading: DeltaNet as explicit-Euler, Longhorn as implicit-midpoint)" | §§12.2–12.3 deliver the pair — with a correction: the proximal stationarity evaluates the gradient at the *endpoint*, so Longhorn is **backward Euler**, not implicit midpoint; ch11's line amended in the same PR. ch11 §11.7 also promised Ch 9's "delta-rule lineage" half on its behalf |
| `ch03:109` | "Chapter 12's delta-rule lineage (DeltaNet's state update is a rank-1 SVD correction)" | §12.2 develops the rank-1 / projector structure of $(I-\beta kk^\top)$ |
| `ch03:218-222` | gave the DeltaNet update verbatim as the recurring "low-rank correction" pattern (Sherman–Morrison context) | §12.2 cites back; keep the algebra consistent with ch03's form |

## 2. Backward-reference anchors (→ `<XRef>` targets)

All confirmed live in `labels.json`:

| Target id | What it is | Where this chapter leans on it |
|---|---|---|
| `ch11:linattn-capacity` | Thm 11.4 (capacity wall) | §12.1 entry motivation |
| `ch11:recurrent-parallel-equivalence` | Thm 11.1 | chunkwise section (same recurrent↔parallel move) |
| `ch11:gla-ltv-duality` | Thm 11.2 | gating section (decay-mask vocabulary) |
| `ch11:fig:mqar-capacity` | Fig 11.4 | if Ch 12 re-runs MQAR with the delta-rule write |
| `ch09:ssd-duality` | Thm 9.5 (SSM ≡ masked linear attention) | unified two-limit view / B teaser |
| `ch06:be-a-stable` | Thm 6.1 (implicit-method stability) | stability section — Longhorn inherits unconditional stability |

## 3. Predecessor reuse

- **High reuse (port):** `post_transformers/experiments/jax/week12/` — complete, tested:
  - `delta_rule.py` (250 LOC) — recurrent DeltaNet primitive + naive-loop oracle;
  - `longhorn.py` (256) — implicit one-step closed-form solve (amortized online learner);
  - `chunkwise.py` (334) — chunkwise DeltaNet (within-chunk parallelism);
  - `stability_analysis.py` (202) — analytic spectral-radius formulas, explicit-vs-implicit;
  - `figures.py` (136) — two-panel spectral-radius comparison; 816 LOC of tests as oracle source.
- **Greenfield (author from paper math):** Gated DeltaNet (W13 is a 4-line stub) and
  Kimi Linear / KDA — from arXiv 2412.06464 and 2510.26692 + dossier.
- **Reference-only (anchor, don't copy):** `linear_recurrence/dossier/03_delta_rule.md`;
  FWP survey 2508.08435; unified implicit attention 2405.16504.

## 4. Bibliography adds

- **Present:** `yang2024gla`, `katharopoulos2020transformers`, `sun2023retnet`, `qin2024hgrn2`,
  `arora2024zoology`, `dao2024mamba2`, `gu2024mamba` — the Ch 11 gating/attention stack.
- **To add (verify authors vs arXiv first):** DeltaNet-parallelization (Yang et al.,
  2406.06484), Gated DeltaNet (Yang et al., 2412.06464), Longhorn (Liu et al., 2407.14207),
  Kimi Linear (2510.26692). Candidate: Schlag et al. 2021 fast-weight programmers (the
  delta-rule-in-transformers origin) — decide during drafting.
- **Defer to a later chapter:** RWKV-7 (2503.14456) and xLSTM keys → Ch 13; hybrid keys → Ch 14.
- **gitleaks watch:** new bibkeys with entropy > 3.5 → append printed
  `ch12-delta-rule-lineage.mdx:generic-api-key:LINE` to `.gitleaksignore`, re-commit.

## 5. Scope tensions / boundaries

- **Convention clash (must resolve in §12.1):** ch11 §11.7 wrote the delta rule with
  $S \in \mathbb{R}^{d_k \times d_v}$ (projector multiplies on the LEFT); ch03 §"low-rank
  correction" and all of `week12/` use $S \in \mathbb{R}^{d_v \times d_k}$ (projector on the
  RIGHT, retrieval $Sk$). Transpose-equivalent, but Ch 12 must pick one (recommend the
  week12/ch03 right-projector form, retrieval $Sk$) and reconcile ch11's form in a margin note
  — otherwise claim-skeptic will (correctly) flag it.
- **Ch 13 hand-off:** RWKV-7's *generalized* delta rule and xLSTM matrix memory get one
  forward-promise paragraph, no content.
- **Kimi depth:** architecture-summary level (KDA structure, 3:1 KDA:full-attention ratio,
  production status as of May 2026 — date-stamp it); no full derivation.
- **B teaser only:** singular-perturbation / two-timescale framing closes the chapter as a
  pointer to Ch 14/16; the methodology lives there.

## 6. Decisions for `/exploring-options` — RESOLVED 2026-06-10 (4 questions, all recommendations accepted)

1. **Companion languages → JAX + torch + Julia**, with Julia scoped to the stability/
   spectral-radius module only (stdlib: LinearAlgebra + Test; joins `companion-julia-tests`).
2. **Section allocation → lens-first arc**: ODE → DeltaNet → Longhorn → stability → chunkwise
   → gating/lineage (the §7 sketch below is confirmed).
3. **Lens explicitness → ODE-first**: §12.1 opens with the associative-recall loss + gradient
   flow; architectures are introduced as discretization choices.
4. **Scope boundary → strict hand-offs**: RWKV-7/xLSTM = forward-promise paragraphs (Ch 13);
   Kimi Linear = architecture summary, date-stamped "as of May 2026"; B framing = closing teaser.

## 7. Likely chapter shape (sketch — confirm against §6 answers)

1. §12.1 The online-learning lens: associative-recall loss, gradient flow, the delta rule
   (capacity-wall entry from `ch11:linattn-capacity`; convention fixed here)
2. §12.2 DeltaNet: one explicit gradient step (rank-1 projector structure → ch03 callback;
   Householder/reflection reading)
3. §12.3 Longhorn: the implicit step (closed-form regularized solve; amortized online learning)
4. §12.4 Stability regions: spectral-radius analysis, explicit-vs-implicit (ch06 XRef; the figure)
5. §12.5 Chunkwise parallelization: recurrent ↔ chunk-parallel equivalence (ch11 Thm 11.1 echo;
   WY-style representation)
6. §12.6 Gating and the production lineage: Gated DeltaNet (decay + delta), Kimi Linear / KDA;
   the unified two-limit view (ch09 SSD XRef) → B teaser
7. §12.7 What's next → §12.8 Exercises → §12.9 Full solutions → §12.10 Companion code

## 8. Companion plan

- **JAX** (`companions/ch12/jax/`): ports of `delta_rule` / `longhorn` / `chunkwise` /
  `stability_analysis` (idiomatic `lax.scan`, float64, port-credit docstrings) + figure
  scripts → `public/figures/ch12/`; greenfield `gated_delta.py` (decay×projector step).
- **torch** (`companions/ch12/torch/`): mirror modules, eager loops, parity vs JAX `<1e-9`.
- **Julia** (`companions/ch12/julia/`): stability sweep only, stdlib (pending Q1).
- **Tests:** loop-oracle ≡ scan `<1e-12`; chunkwise ≡ recurrent `<1e-12`; analytic
  spectral-radius formulas pinned; Longhorn unconditional-stability assertion (float-safe form);
  parity `<1e-9`; `--import-mode=importlib`.

## 9. Gate items + gotchas

- Companions FIRST; captions quote measured stdout numbers (F19).
- claim-skeptic + chapter-auditor before `status: implemented`; the §5 convention clash is the
  known claim-skeptic tripwire.
- Honor the exercise-heading format (`### Exercise 12.M …` / `### Solution to Exercise 12.M`).
- Makefile: add `companions/ch12/torch` to `companion-torch-tests` (+ julia loop if Q1 says yes);
  verify the Edit applied.
- Branch `feat/ch12-delta-rule-lineage`; gitleaks at commit time; explicit push; PR → CI →
  merge-commit → deploy; then the full doc-sync checklist.
