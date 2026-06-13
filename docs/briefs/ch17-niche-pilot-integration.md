# Chapter research brief — Ch 17: Niche-pilot integration

> The LAST chapter (M6 = book content-complete). Decisions §6 resolved 4/4 on 2026-06-13.
> This brief is the durable record the `chapter-auditor` / `claim-skeptic` read against.

- **Chapter / slug:** ch 17 — `ch17-niche-pilot-integration.mdx` (stub exists; title
  `'Niche-pilot integration: from curriculum to research output'` — **preserve verbatim**)
- **Part / status target:** synthesis → `implemented`
- **One-line scope:** the crown-jewel chapter — how the 17-chapter curriculum maps onto the
  research pilots (C1 symplectic, B two-timescale) and the broader 13-niche portfolio; ships the
  *integrated template* (instruments composed end-to-end on idealized systems) + the niche decision
  rubric. Idea a reader leaves with: **the dynamical-systems lens is a research instrument, and here
  is how the book's pieces compose into one.**
- **Pilot tie-in:** BOTH (C1 + B) — this is the integration chapter. But the pilots are *in-progress*
  (results pending Q3 2026), so Ch 17 ships the template, not trained-model findings.

## 1. The defining constraint (shapes everything)

The pilots' empirical results **do not exist yet** — `experiments/jax/week20/crown_jewel.py` is a
5-line stub, the C1 `symplectic_atlas/` and B `twotimescale/` code dirs aren't created. So Ch 17
**cannot report trained-model results**. It redeems the "application" forward-promises the same
honest way ch14/15/16 did ("trained-model probing is the pilot's program"): compose the instruments
end-to-end on the book's *idealized/constructed* systems → real measured **integration signatures**
(the reproducible template) + curriculum→pilot map + decision rubric, empirical results forward-ref'd.

## 2. Forward-promises to redeem

| Source (file:line) | Promise | How Ch 17 honours it |
|---|---|---|
| ch06:164,174 | C1 applies the modified-Hamiltonian framework to learned selective dynamics | §17.2 atlas-cell demo (idealized oscillator SSM mode); empirical part → C1 pilot |
| ch10:43,182,272 | revisits the discretization stability atlas | §17.2 the worked atlas cell (exact-exp vs symplectic vs RK4 energy drift) |
| ch14:§14.6/§14.7 | the two-timescale seed + matched-decay as B's foundation | §17.3 builds the B pipeline on the ch14 HMM |
| ch15:58,344,348–357 | the synthesis: turn Ch 15's diagnostics + Ch 16's protocol on C1/B; weigh counter-evidence | §17.3 end-to-end pipeline; §17.5 the provisional verdict vs the capacity ceiling |
| ch16:572–573 | what C1/B used; what the lens earned | §17.4 the curriculum→niche map; §17.5 verdict |

## 3. Backward-reference anchors (→ `<XRef>`, confirmed in `labels.json`)

C1: `ch06:symplectic-modified-hamiltonian`, `ch06:fig:energy-drift`, `ch06:be-a-stable`,
`ch10:exp-trap-order`, `ch10:exp-trap-stability`, `ch10:rope-complex-equivalence`,
`ch10:fig:complex-spiral`. B: `ch14:matched-decay-optimal`, `ch14:slow-manifold-tracking`,
`ch15:capacity-bound`, `ch15:lyapunov-estimator`, `ch15:regime-separation`,
`ch16:discriminative-regime`, `ch16:paired-comparison`, `ch16:probe-recoverability`. Lens:
`ch01:def:matexp`, `ch02:def:lyap`, `ch13:generalized-spectrum`.

## 4. Predecessor reuse + in-book reuse

- **Predecessor: NONE runnable.** `week20/crown_jewel.py` is a 5-line stub (verified). The C1/B
  pilot code dirs don't exist yet. Reference-only: `notes/foundations_curriculum_design_2026_05_20.md`
  §21 (Ch 17 scope), `notes/niche_decision_2026_05_24.md` (C1+B), the C1/B kickoff docs.
- **In-book reuse (the integration companions compose these — verified APIs):**
  - C1: `companions/ch06/jax/symplectic_demo.py` (`harmonic_H`, `harmonic_T_grad`/`harmonic_V_grad`,
    `verlet_step`, `rk4_step_hamilton`, `simulate`, `rk4_drift_per_period`);
    `companions/ch10/jax/complex_state.py` (`complex_scalar_recurrence`, `decay_rate`).
  - B: `companions/ch14/jax/two_timescale.py` (`TwoTimescaleHMM`, `make_hmm`, `make_transition`,
    `forward_filter_predictions`, `decay_filter_predictions`, `window_filter_predictions`,
    `unigram_filter_predictions`, `mean_cross_entropy`, `epsilon_to_lambda`);
    `companions/ch16/jax/protocol.py` (`reference_instance`, `composite_filter_predictions`,
    `probe_signature`, `paired_comparison_stats`, `filter_regime_priors`);
    `companions/ch15/jax/lyapunov_diagnostics.py` (`effective_state_size`, `lyapunov_spectrum`).
  - `companions/_shared/plot_utils.py` (`save_figure`, `create_tufte_figure`, `apply_style`,
    `SSM_COLORS`, `set_tufte_labels/title`).

## 5. Bibliography adds (+0)

`grep -c '^@'` = 67, expected 67 after. ALL anchors PRESENT (verified): `hairer2006geometric`,
`lahoti2026mamba3`, `kokotovic1986singular`, `poli2024mechanistic`, `arora2024zoology`,
`anonymous2025lyapunov`, plus the ch15 counter-evidence keys. **No new entries** — a synthesis
chapter cites existing references. (gitleaks watch: any high-entropy `<Cite key=...>` → append the
printed `ch17-...:generic-api-key:LINE` to `.gitleaksignore`.)

## 6. Decisions for `/exploring-options` — RESOLVED 2026-06-13 (4/4 recommendations accepted)

1. **Status `implemented` + two thin integration companions** — each composes existing instruments
   into an end-to-end demo on idealized systems producing a NEW integrated signature (not a re-run).
2. **Two-pilot spine** — C1 then B (chapters-drawn-from → instrument stack → idealized demo →
   empirical forward-ref), framed by the lens-earned intro + niche rubric close.
3. **C1+B deep + 13-niche survey** — the active pilots in depth + a survey map of the 13 niches +
   a niche-selection rubric.
4. **No new theorems; integrative exercises** — synthesis composes; backward-XRef prior props.

## 7. Likely chapter shape (sketch — STYLE §13 positional accommodation, Ch 5 precedent)

1. §17.1 What the dynamical-systems lens set out to earn (framing; cited-vs-demonstrated + idealized-vs-trained boundaries)
2. §17.2 C1 — symplectic integrators (atlas-cell demo; `c1-atlas-cell.png`; empirical forward-ref)
3. §17.3 B — two-timescale benchmarks (end-to-end pipeline; `b-disentanglement.png`; weigh vs ch15 ceiling)
4. §17.4 The 13-niche portfolio and the decision rubric (survey map + rubric table)
5. §17.5 What the lens earned, and its limits (provisional verdict; the capacity ceiling boundary)
6. §17.6 Where this goes (pilots' empirical execution; post-M6 beta gate; book content-complete)
7. §17.7 Exercises (3 short + 3 long, integrative) → §17.8 Full solutions → §17.9 Companion code

## 8. Companion plan

- **JAX** (`companions/ch17/jax/`):
  - `c1_integration.py` — the C1 atlas cell. Compose ch06 (Verlet/RK4 energy drift) + ch10
    (complex-mode SSM). NEW signature: long-horizon energy drift of {exact-exponential SSM transition,
    symplectic Verlet, non-symplectic RK4} on a harmonic-oscillator SSM mode — showing the diagonal
    SSM's exact exponential already conserves the imaginary-mode energy (so symplectic structure
    bites only off the diagonal — the C1 pilot's real question). Figure `c1-atlas-cell.png`.
  - `b_integration.py` — the B pipeline. Compose ch14 (HMM + idealized predictors) → ch16
    (composite predictor, probe_signature, paired stats) → ch15 (effective state size of each
    predictor's implied transition). NEW signature: the integrated disentanglement readout linking a
    predictor's effective state size (ch15) to its regime-recovery probe accuracy (ch16) and CE
    (ch14) across the predictor family. Figure `b-disentanglement.png`.
- **Julia** (`companions/ch17/julia/`): `symplectic_crosscheck.jl` — stdlib symplectic
  energy-conservation cross-check of the C1 atlas cell (echoes ch10's Julia + the C1 pilot's Julia
  `symplectic_atlas`); JAX↔Julia parity `<1e-9` on the shared energy-drift anchor.
- **torch**: NONE — the demos compose existing JAX modules; no new kernel → a port would only re-run.
- **Tests** (~25–35 JAX + ~8 Julia): each integrated signature pinned at printed precision; each
  reduction-to-component pinned (`rtol=0`) — C1: RK4 drift ≡ ch06 `rk4_drift_per_period`, exact-exp
  conserves to 1e-15; B: composite ≡ ch16 at the operating point, probe ≡ ch16, ESS ≡ ch15;
  JAX↔Julia energy parity `<1e-9`; validation raises.

## 9. Gate items + gotchas

- All four review subagents before `status: implemented`. **Claim-skeptic pressure points:** (1)
  each integration signature must be NEW (headline = new measurement; reductions = consistency
  checks, not the claim — the Ch 15 duplication-trap lesson); (2) **no trained-model results** (every
  demo idealized; the empirical program forward-ref'd); (3) the verdict is *provisional* (results
  pending); (4) the 13-niche taxonomy attributed to the curriculum design doc (no overclaim).
- Companions-first (prose cites measured numbers). `npm run build` is the only MDX compiler.
  **Every inline `$...$` on ONE line**; **unquoted `description:` must not contain `: `** (the ch15
  build break); matplotlib mathtext ≠ KaTeX (`\*` invalid). No `<Theorem>` blocks (none added);
  backward-XRefs must resolve.
- After shipping (= M6): doc-sync IN the PR; roadmap + playbook memory (campaign COMPLETE 6/6, 11×);
  file the M6 notification issue to post_transformers (mirrors M3/issue #53).
