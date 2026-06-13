# Chapter research brief — Ch 13: Exponential gates and matrix memory (xLSTM, RWKV-7)

> Filled 2026-06-12 from the Ch 13 pre-recon memory + same-day verification against the
> shipped chapters and the ch12 companion API. Drives the `/exploring-options` round (§6).
> Campaign order after M3: **13 → 15 → 17**. Recipe: the chapter-authoring playbook
> (proven 8×: 7–12, 14, 16).

- **Chapter / slug:** ch 13 — `ch13-exponential-gates-matrix-memory.mdx` (stub exists, `status: planned`)
- **Part / status target:** beyond-ssm → `implemented`
- **One-line scope:** the generalized linear recurrence whose transition is **diagonal-plus-rank-one** —
  RWKV-7's generalized delta rule (rank-one removal in a *learned* direction, not the key) and xLSTM's
  matrix memory with **exponential gating** (and the log-domain stabilizer state that overflow forces) —
  extending Chapter 12's delta-rule lineage to matrix memories with their own stability questions. The idea
  a reader leaves with: *moving the gate inside the nonlinearity buys expressivity but imports a numerical
  stability problem the architecture must solve in its own state.*
- **Pilot tie-in:** **none direct** — C1 was satisfied by Ch 10, B unblocked by Ch 16. But the curriculum
  (§17) lists Ch 13 as "required by A3 (xLSTM as Lyapunov target), A4 (regime detection)" — both are *Ch 15
  diagnostics* concerns. Ch 13 ships the architecture-level stabilization *mechanism*; the *diagnostics on
  trained models* are Ch 15. So Ch 13 is a stability-thread chapter that feeds Ch 15, not a pilot anchor.

## 1. Forward-promises to redeem

All five quoted and verified 2026-06-12 (`grep -rn "Chapter 13" src/content/chapters/`).

| Source (file:line) | Promise made | How this chapter honours it |
|---|---|---|
| `ch12:421–429` | **PRIMARY**: "Chapter 13 takes the lineage's next generalization: RWKV-7's *generalized* delta rule (transition $\mathrm{diag}+$ rank-one in a learned direction, not the key itself) and xLSTM's matrix memory with exponential gating — the gate moved inside the nonlinearity, with its own stabilization problem." | The whole chapter. §13.2 generalized transition; §13.3 RWKV-7; §13.4 xLSTM exp-gate + stabilizer. |
| `ch12:171` | "exactly where Chapter 13's matrix-memory architectures … pick up" | §13.1 framing (picks up the capacity-vs-staleness thread). |
| `ch14:354–357` | "xLSTM's exponential gates (granularity: vector; trigger: input; plus a stabilizer state the sigmoid families do not need) and the test-time memory writes of the **Titans line** (trigger: the *loss*, a fourth trigger class) are Chapter 13's subjects, where their matrix-memory context lives." | §13.4 (xLSTM vector gates + stabilizer); §13.5 (Titans = the fourth trigger class, taxonomy depth). |
| `ch14:533–535` | "Chapter 13 lives inside the gates themselves: exponential gating with its stabilizer states (xLSTM) and the generalized delta rule (RWKV-7) extend Chapter 12's lineage to matrix memories with their own stability questions." | The §13 spine: gates-inside, stabilizer, generalized transition, stability. |
| `ch16:567–568` | "Chapter 13 returns to architecture, inside the gates themselves: exponential gating and matrix memories (xLSTM, RWKV-7) extending Chapter 12's lineage." | Confirms the architecture (not methodology) framing. |

## 2. Backward-reference anchors (→ `<XRef>` targets)

All confirmed resolving in `labels.json` 2026-06-12 (id corrections applied).

| Target id | What it is | Where this chapter leans on it |
|---|---|---|
| `ch12:stability-dichotomy` | explicit/implicit delta-rule stability dichotomy | §13.2 — **template** for the generalized-transition stability prop (P1) |
| `ch12:delta-rule-two-forms` | the delta rule's recurrent/chunkwise two forms | §13.3 — RWKV-7 generalizes; reduction prop (P3) |
| `ch12:wy-representation` | chunkwise WY representation | §13.3 — parallel form for the generalized transition |
| `ch12:longhorn-closed-form` | Longhorn implicit-step closed form (unconditional stability) | §13.4 — contrast: implicit stability vs exp-gate stabilization |
| `ch12:recall-fixed-point` | the associative-recall fixed point | §13.x — matrix-memory recall (no re-proof) |
| `ch14:gate-granularity` | the gate-granularity proposition | §13.4 — places xLSTM's vector exp-gates in the taxonomy |
| `ch11:linattn-capacity` | additive-state capacity wall $\sqrt{K/d}$ | §13.x — matrix-memory capacity discussion (cite, no re-proof) |
| `ch09:ltv-transition` | LTV transition operator | §13.2 — the generalized transition is an LTV recurrence |
| `ch09:ssd-duality` | selective SSM ≡ masked linear attention | §13.2 — diagonal decay generalizes SSD's scalar gate |
| `ch06:be-a-stable` | implicit-method A-stability | §13.4 — the stabilizer-as-numerical-stability framing |
| `ch05:dahlquist-barrier` | (optional) the stability thread | §13.2/§13.4 if the explicit-step analogy is invoked |
| Ch 2 (BIBO / bounded state) | — *no XRef-able label exists* | §13.4 — **prose reference only** for the bounded-state reading of the stabilizer |

## 3. Predecessor reuse

- **High reuse (in-book, EXTEND — do not re-derive):** `companions/ch12/jax/gated_delta.py` and
  `stability.py`. RWKV-7's transition $\mathrm{Diag}(w_t) - \beta_t a_t a_t^\top$ is *exactly* a
  generalization of `gated_delta_step`'s $\gamma_t(I-\beta_t k_tk_t^\top)$ (scalar $\gamma_t \to$ diagonal
  $w_t$; key-direction $k_t \to$ a **learned** direction $a_t$). The companion imports `gated_delta_step`
  and pins the reduction (P3). `stability.py`'s spectral-radius pattern (`deltanet_spectral_radius`,
  `iteration_eigenvalue_along_k`, the Rayleigh-quotient guard) is the template for P1.
- **Greenfield (author from paper math):** the xLSTM **mLSTM** matrix-memory recurrence
  $C_t = f_t C_{t-1} + i_t v_t k_t^\top$ with **exponential** input/forget gates, the normalizer state
  $n_t$, and the log-domain **max-state stabilizer** $m_t$ (Beck et al.); the RWKV-7 generalized delta-rule
  transition (Peng et al.); the unifying generalized-transition object that frames both.
- **Predecessor code: NONE** — `post_transformers/experiments/jax/week14/xlstm.py` and `week15/rwkv7.py`
  are **4-line TODO stubs** (verified 2026-06-12; their `test_*.py` are stubs too). Greenfield companions,
  like ch14/ch16.
- **Reference-only (anchor, don't copy):** `post_transformers/references/dossier/gating_design_space/`
  (trigger/granularity rubric incl. the Basu finding — already mined for ch14); curriculum
  `notes/foundations_curriculum_design_2026_05_20.md` §17 (Ch 13 entry: sLSTM+mLSTM, exp gating, RWKV-7
  generalized delta rule, production xLSTM 7B / RWKV-7 2.9B, "A3 xLSTM-as-Lyapunov-target / A4
  regime-detection" → Ch 15).

## 4. Bibliography adds

`bibliography.bib` has **61** entries and **zero** xLSTM/RWKV/Titans keys (verified).

- **Present:** the delta-rule lineage (`yang2024gated`, Kimi/KDA), `arora2024zoology`, `arora2024simple`,
  the gate-granularity sources — all reusable for context, none cover xLSTM/RWKV/Titans.
- **To add (+3, first-word-of-title bibkey rule — ch16 C1 lesson; authors + ids verified via arXiv API
  at add time):**
  - `beck2024xlstm` — xLSTM: Extended Long Short-Term Memory, Beck et al. (NeurIPS 2024). arXiv:2405.04517 — **VERIFY**
  - `peng2025rwkv` — RWKV-7 "Goose" with Expressive Dynamic State Evolution, Peng et al. (2025). arXiv:2503.14456 — **VERIFY**
  - `behrouz2025titans` — Titans: Learning to Memorize at Test Time, Behrouz et al. arXiv:2501.00663 — **VERIFY year/venue** (cited even at summary depth, per §5).
- **Defer:** Lyapunov/regime-detection diagnostics theory → Ch 15.
- **gitleaks watch:** new bibkeys + math-heavy `$...$` lines may trip entropy > 3.5 → append printed
  `chNN-slug.mdx:generic-api-key:LINE` fingerprints to `.gitleaksignore`, never `--no-verify`.

## 5. Scope tensions / boundaries

- **Titans depth (→ §6 Q4).** ch14:354 pre-committed that the Titans line's "matrix-memory context lives"
  here. But Titans is test-time-training flavored (gradient-at-inference memory writes) — a full companion
  balloons scope. **Recommend taxonomy/summary depth:** name it as the *fourth trigger class* (loss-triggered
  write), place it in the gate-trigger taxonomy completing ch14's input/output/data picture, forward-ref;
  **no companion.**
- **Ch 15 boundary (strict).** Stability *diagnostics on trained models* — Lyapunov exponents, ESS,
  regime detection (the curriculum's "A3 xLSTM-as-Lyapunov-target", "A4 regime detection") → **Ch 15**.
  Ch 13 owns the architecture-level stabilization *mechanism* (the $m_t$ normalizer, decay clamps, the
  transition spectrum). Forward-ref Ch 15 for the diagnostic complement.
- **ch12 overlap.** Do **not** re-derive the delta-rule core — XRef ch12 and *extend* its companions. Ch 13's
  value is the **generalized transition** (diag + rank-one in a learned direction), **exp-gate stabilization**,
  and the **matrix-memory** framing — not a re-presentation of DeltaNet.
- **sLSTM vs mLSTM.** xLSTM has two cells: sLSTM (scalar memory, memory-mixing) and mLSTM (matrix memory,
  fully parallelizable). **Name both; companion + propositions focus on mLSTM** (the title's "matrix memory")
  and the **shared exponential-gate stabilizer.** sLSTM gets prose treatment, not a companion.
- **Cross-repo links:** dossier/curriculum provenance via absolute GitHub URLs pinned to main (CLAUDE.md).

## 6. Decisions for `/exploring-options` — RESOLVED 2026-06-12 (4/4 recommendations accepted)

1. **Companion languages: JAX + torch + a small Julia stability module.** JAX canonical, torch parity
   (0527-F27), **and** `companions/ch13/julia/xlstm_stabilization.jl` — naive exp-gate overflow vs the
   log-domain max-state stabilizer, stdlib-only, pinning the exactness identity. The stabilizer is the
   chapter's signature numerical-stability content, squarely in the book's selective-Julia-for-stability lane
   (ch04/05/06/12); the `ch12/julia/delta_stability.jl` template applies.
2. **Arc: unification-first.** The generalized linear transition $\mathrm{Diag}(w_t)-\beta_t a_ta_t^\top$ is
   the spine; xLSTM and RWKV-7 are two realizations; stability = the transition spectrum. (Outline in §7.)
3. **Theorems: three propositions**, all provable in-page, all numbers companion-measured:
   (P1) **generalized-transition stability spectrum** — radius of $\mathrm{Diag}(w_t)-\beta a a^\top$,
   generalizing ch12's $\rho_k=|1-\beta\|k\|^2|$; (P2) **exp-gate stabilizer exactness** — the log-domain
   max-state recurrence equals the naive exp recurrence's normalized readout wherever the latter doesn't
   overflow (exact, not approximate); (P3) **reduction** — RWKV-7's generalized delta rule recovers ch12's
   gated DeltaNet when the learned direction is the key and the diagonal is scalar.
4. **Titans scope: taxonomy/summary subsection** (the fourth trigger class), **no companion.**

## 7. Likely chapter shape (sketch)

1. **§13.1 The move: the gate inside the nonlinearity** — motivation; the generalized linear recurrence;
   capacity-vs-staleness pickup (XRef ch11/ch12); what "matrix memory" buys.
2. **§13.2 The generalized transition** — $\mathrm{Diag}(w_t) - \beta_t a_t a_t^\top$; **P1 stability
   spectrum** (XRef ch12:stability-dichotomy, ch09:ltv-transition); figure: spectrum vs ch12's scalar-gate case.
3. **§13.3 RWKV-7's generalized delta rule** — rank-one removal in a *learned* direction; chunkwise/WY form
   (XRef ch12:wy-representation); **P3 reduction to ch12's gated DeltaNet**; pinned reductions (or small figure).
4. **§13.4 xLSTM: matrix memory and exponential gating** — mLSTM recurrence + exp input/forget gates; the
   overflow problem; the log-domain max-state stabilizer ($m_t$, $n_t$); **P2 stabilizer exactness**; sLSTM
   named; figure: naive-overflow vs log-domain-stable. XRef ch14:gate-granularity, ch06:be-a-stable; Ch 2 prose.
5. **§13.5 Stability questions, the trigger taxonomy, and the production lineup** — the stabilizer as a
   bounded-state mechanism; **Titans = the fourth trigger class** (summary, no companion); forward to Ch 15
   diagnostics; production status (xLSTM 7B, RWKV-7 2.9B).
6. **§13.6 What's next** → **§13.7 Exercises** (3 short + 3 theory) → **§13.8 Full solutions** → **§13.9 Companion code.**

## 8. Companion plan

- **JAX** (`companions/ch13/jax/`):
  - `generalized_transition.py` — the $\mathrm{Diag}(w_t)-\beta_t a_t a_t^\top$ transition; spectral radius
    (P1) + Rayleigh-quotient guard (extend `stability.py` pattern); RWKV-7 generalized delta rule recurrent +
    Python-loop naive oracle; chunkwise form; **P3 reduction** importing `companions.ch12.jax.gated_delta`.
    → `transition-spectrum.png`.
  - `xlstm.py` — mLSTM matrix-memory recurrence with exponential gates; the **naive exp recurrence** (overflows)
    vs the **log-domain max-state stabilizer** ($m_t$, $n_t$ normalizer); **P2 exactness** identity (stabilized
    == naive where finite, to `<1e-12`); overflow demonstrated (naive → Inf/NaN). → `stabilizer-overflow.png`.
  - (Optional third / fold-in) recall-capacity demo vs ch11/ch12 capacity wall → `matrix-memory-capacity.png`.
- **torch** (`companions/ch13/torch/`): mLSTM + generalized-transition scoring-path mirrors; in-process JAX
  parity `<1e-9` (tokens/keys drawn once, fed to both). No figures.
- **Julia** (`companions/ch13/julia/`, **only if §6 Q1 = yes**): `xlstm_stabilization.jl` — naive exp overflow
  vs log-domain stabilizer, stdlib-only, pin the log-domain == naive identity in the finite regime;
  `runtests.jl`. (`ch12/julia/delta_stability.jl` template.)
- **Tests** (STYLE.md §8 bar): ~40–55 JAX + ~6–10 torch (+ ~15–25 Julia if approved); exact identities
  `<1e-12` `rtol=0` (P2 exactness, P3 reduction); JAX↔torch parity `<1e-9` float64; independent oracles
  (Python-loop for the transition; closed-form for the spectrum); every figure's load-bearing quantity pinned
  at printed precision; `--import-mode=importlib` (no `__init__.py`).

## 9. Gate items + gotchas

- Companions-first; prose cites only **measured** numbers from companion stdout (F19 caption-truthfulness).
- Every inline `$...$` span on **ONE physical line** (`- `/`+ ` wraps break acorn at `npm run build`); run the
  odd-`$`-count line scan **and** `npm run build` before push (validate doesn't compile MDX).
- **Never hard-code theorem/proposition numbers** — XRef self-refs; `labels.json` counts propositions; no XRef
  inside figure captions.
- Prop hypotheses must cover *exactly* what the companion implements (claim-skeptic, ch16 lessons): P1's
  stability condition must match the implemented spectrum; P2's "exact where finite" must be the pinned
  identity, not an approximation claim; exercise answers **measured + pinned, never estimated**.
- All four review subagents pre-ship (claim-skeptic + chapter-auditor + companion-verifier +
  prose-pedagogy-reviewer); fix every finding before `status: implemented`.
- **Pre-commit shim (2026-06-11):** the repo `.venv/bin/pre-commit` must exist or commit **and** push silently
  block — `uv pip install pre-commit` into the repo .venv; never `--no-verify`, never overwrite the user's shim.
- torch tests via the **Makefile target** (no `PYTHONPATH=.`); JAX mains need `PYTHONPATH=.`.
  Julia (if approved): stdlib-only, juliaup-lock recovery if the toolchain wedges.
- Doc-sync rides **IN** the PR (ch14 C1 lesson): `CLAUDE.md` status line, `README` banner+row, `docs/DASHBOARD.md`,
  regen `docs/STATUS.md`, `CURRENT_WORK.md` (next = **Ch 15**).
- Ship: branch `feat/ch13-exponential-gates`, stage chapter artifacts only, explicit `git push -u`.
- After shipping: roadmap memory (Ch 13 done; NEXT = Ch 15), playbook lessons, `docs/DASHBOARD.md`.
  (No post_transformers issue needed — no pilot milestone rides on Ch 13.)
