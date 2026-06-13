# Chapter research brief — Ch 15: Counter-evidence and diagnostic tools

> Filled before authoring (the `/exploring-options` input). Decisions in §6 resolved
> 4/4 on 2026-06-13; this brief is the durable record the `chapter-auditor` /
> `claim-skeptic` read against.

- **Chapter / slug:** ch 15 — `ch15-counter-evidence.mdx` (stub exists; frontmatter title
  `'Counter-evidence and diagnostic tools: where SSMs fail'` — **preserve verbatim**)
- **Part / status target:** integration → `implemented`
- **One-line scope:** the prosecution's file — what fixed-state / SSM-heavy designs *provably
  cannot* do (the TC⁰ ceiling, the copying separation, the illusion of state) and the
  *diagnostic toolkit* (Lyapunov exponents, regime detection, effective state size) that
  measures stability/capacity on real systems. The idea a reader leaves with: **an
  architecture's promise is an asymptotic ceiling; a diagnostic is what tells you where a
  concrete system sits beneath it.**
- **Pilot tie-in:** none directly (C1 closed at Ch 10, B's book-side prereqs closed at Ch 16).
  But Ch 15 *builds the instruments pilot B uses* on trained models — so the no-training
  discipline and the forward-ref to B are load-bearing.

## 1. Forward-promises to redeem

| Source (file:line) | Promise made | How this chapter honours it |
|---|---|---|
| `ch13:61` (NoteBox) | diagnostics (Lyapunov, regime detection) probe matrix-memory stability **on trained models** | §15.4–15.5 build the diagnostics on constructed systems; forward-ref B for the trained-model run |
| `ch13:220–221` (§13.3 margin) | RWKV-7's "recognizes all regular languages" claim **placed against impossibility** | §15.2 cited-results box places it against the TC⁰ ceiling |
| `ch13:326–327` (§13.5) | the two stability registers measured on **trained** networks | §15.4 (Lyapunov) + §15.5 (regime/effective size); trained run = B |
| `ch13:356–361` (§13.6) | **the counter-evidence file**: stability → diagnostics + impossibility (TC⁰); the generalized transition + stabilizer state are the objects the diagnostics probe | whole chapter; DPLR transition reused as a constructed diagnostic target (§15.4) |
| `ch16:138–139` (§16.2) | "the impossibility side … is **Chapter 15's file**" | §15.3 (capacity bound, P1′) + §15.2 (cited TC⁰) |
| `ch16:569–571` (§16.7) | "the prosecution's file … and the diagnostic toolkit for catching it" | whole chapter; §15.3 redeems the §16.2 copying-probe hand-off |
| `ch14:537–542` (§14.7) | "**Chapter 15** plays prosecution … the diagnostic toolkit" | whole chapter; §15.6 ties G&A to ch14's hybrid verdict |
| `ch07:148` (§7.4) | Lyapunov analysis of a trained S4 layer **vs the HiPPO spectrum** | §15.4 worked example uses an S4D-resembling diagonal spectrum (decision 4) |
| `ch02:202` (§2.5) | **eigenvalue tracking during training** as a unified Lyapunov + BIBO diagnostic | §15.4 builds on `ch02:def:lyap` + the `qr_lyapunov` engine; the BIBO link recalled |

## 2. Backward-reference anchors (→ `<XRef>` targets, confirmed in `labels.json`)

| Target id | What it is | Where this chapter leans on it |
|---|---|---|
| `ch02:def:lyap` | Lyapunov-exponent definition | §15.4 (the estimator validates this object) |
| `ch02:fig:lyapunov-spectrum` | QR-spectrum figure | §15.4 (the engine reused; the new systems are LTV/DPLR) |
| `ch02:thm:bibo-tf` | BIBO via transfer function | §15.4 (the unified stability diagnostic) |
| `ch07:*` (HiPPO spectrum) | trained-S4 spectrum reading | §15.4 worked example resemblance |
| `ch11:linattn-capacity` | rank capacity wall | §15.3 (P1′ generalizes it — *instance*, not re-derivation) |
| `ch13:generalized-spectrum` | DPLR transition spectrum | §15.4/§15.5 (constructed diagnostic targets) |
| `ch13:rwkv-reduction` | RWKV-7 ↔ gated DeltaNet | §15.2 (the regular-language claim's object) |
| `ch16:discriminative-regime` | slot model `min(1,d/N)` | §15.3 (P1′'s measured shadow — *instance*) |

## 3. Predecessor reuse

- **High reuse (port): NONE that is runnable.** `post_transformers/experiments/jax/week18/lyapunov_ssm.py`
  is an 89-byte **stub** (TODO only) — verified, not reusable. `week12/stability_analysis.py`
  is complete but solves a *different* problem (iteration-matrix spectral radius, not Lyapunov
  on trajectories).
- **In-book reuse (the real anchor):**
  - `companions/ch02/jax/lyapunov_qr.py` — `qr_lyapunov(jacobians, n_steps)` Benettin QR engine
    (**reuse verbatim**); `autonomous_lyapunov_reference` for ground truth.
  - `companions/ch09/jax/selective_ssm.py` — `discretize_selective(A, delta, B)` → `Abar`
    (per-step diagonal transitions = the LTV Jacobian stream); `selective_scan/sequential`.
  - `companions/ch13/jax/generalized_transition.py` — `dplr_transition(w,a,c)`,
    `transition_spectrum(w,a,c)` (the constructed DPLR target + its real spectrum).
  - `companions/ch16/jax/mqar.py` — `make_mqar`, `slot_reader`, `slot_accuracy_exact`,
    `accuracy` (P1′ imports these; the cliff at `N=d` is asserted, not re-implemented).
  - `companions/_shared/plot_utils.py` — `save_figure`, `create_tufte_figure`, `apply_style`,
    `SSM_COLORS`, `set_tufte_labels/title`.
- **Reference-only (anchor, don't copy):** `post_transformers/notes/synthesis.md` §1
  (counter-evidence literature table); `roadmap.md` W10 + W18.

## 4. Bibliography adds (+3; arXiv-verified 2026-06-13)

`grep -c '^@'` = 64 before → 67 after; `npm run build:bib` after edits.

- **Present:** `benettin1980lyapunov`, `jelassi2024repeat`, `arora2024zoology`,
  `olsson2022incontext`, `lee2025understanding` (the Mamba-Transformer-hybrid recall paper,
  **not** G&A), `anonymous2025lyapunov` (TMLR Mamba Lyapunov).
- **To add (`<firstauthor><year><firstword>`):**
  - `merrill2024illusion` — Merrill, Petty, Sabharwal 2024, "The Illusion of State in
    State-Space Models", ICML 2024, arXiv:2404.08819. **The SSM-specific ceiling.**
  - `merrill2023parallelism` — Merrill, Sabharwal 2023, "The Parallelism Tradeoff:
    Limitations of Log-Precision Transformers", TACL 2023, arXiv:2207.00729. **The TC⁰ anchor.**
  - `bick2025understanding` — Bick, Xing, **Gu** 2025, "Understanding the Skill Gap in
    Recurrent Language Models: The Role of the Gather-and-Aggregate Mechanism", arXiv:2504.18574.
    **G&A — co-authored by Mamba's creator (Albert Gu).**
- **Defer:** Arora mechanistic-evaluation (2505.15105), Chen Achilles-heel (2509.17514),
  Engelken RNN-Lyapunov (2006.02427) — mention without keying unless §15.4/§15.6 prose
  actually leans on them.
- **gitleaks watch:** `<Cite key=...>` lines trip `generic-api-key` (entropy > 3.5) → append the
  printed `ch15-counter-evidence.mdx:generic-api-key:LINE` to `.gitleaksignore`, never `--no-verify`.

## 5. Scope tensions / boundaries

- **The duplication trap (the headline design constraint).** A naïve "state-capacity copying
  bound verified by ch16's slot model" *is* `ch16:discriminative-regime` in information units,
  and a cousin of `ch11:linattn-capacity`; a naïve "Lyapunov estimator vs closed form" *is*
  `ch02:lyapunov_qr` + `ch02:fig:lyapunov-spectrum`. **Resolution (it's an integration
  chapter — raise the altitude):** P1′ is stated for *any deterministic finite-state
  recurrence* (architecture-agnostic pigeonhole), of which ch11/ch16 are *instances it
  backward-refs*; P2′ runs on *new* constructed LTV/DPLR systems with the degeneracy hypothesis
  explicit; P3′ (regime separation) is genuinely new, guarded by a two-route cross-check.
- **No-training (strict).** Companions are deterministic — no gradient training, no heavy
  weights dependency. Every diagnostic runs on *constructed* systems with **known
  ground-truth spectra**; one worked example is *sized to resemble* an S4D/selective spectrum
  (decision 4). Trained-model probing is **pilot B's program** (forward-ref, do not do it).
- **TC⁰ is cited, never re-proven.** Deep circuit complexity (Merrill et al.) — stated with
  attribution "of [authors], [year]" so it is never mistaken for our theorem. The one
  self-contained impossibility we *can* prove in-page is the counting bound (P1′), introduced
  explicitly as "weaker than, and the honest shadow of," the cited results.
- **Ch 17 owns synthesis.** Ch 15 lays the evidence + tools; "when to reach for attention vs
  SSM given the prosecution's file" is Ch 17 (pilot integration).
- **Cross-repo links** to post_transformers as absolute GitHub URLs pinned to `main`.

## 6. Decisions for `/exploring-options` — RESOLVED 2026-06-13 (4/4 recommendations accepted)

1. **Companion languages: JAX + torch + Julia** (full triad, like ch13). JAX canonical; torch
   parity (B runs diagnostics on trained *torch* models); stdlib **Julia QR-Lyapunov
   cross-check** (Lyapunov is the headline NA instrument). Skip the trivial torch
   copying-bound port (integer arithmetic — ceremony).
2. **Arc: impossibility-first (prosecution)** — ceiling (TC⁰ cited + P1′) → instruments
   (P2′ Lyapunov correctness + P3′ regime separation on constructed systems) → mechanistic
   verdict (G&A). Matches every forward-ref's "prosecution file" framing.
3. **Theorems: 3 propositions + a cited-results box.** P1′ architecture-agnostic capacity
   bound; P2′ Lyapunov-estimator correctness (degeneracy hypothesis explicit); P3′ regime
   separation via effective state size (two-route anti-circularity cross-check). TC⁰ /
   illusion-of-state / copying cited; G&A a cited remark.
4. **No-training boundary: constructed systems + resemblance framing** (see §5).

## 7. Likely chapter shape (sketch)

1. §15.1 The prosecution's file (the two threads; named forward-refs; proven-vs-cited boundary)
2. §15.2 The expressivity ceiling — TC⁰ + illusion-of-state (cited-results box); RWKV-7 claim placed against it
3. §15.3 The capacity bound — **P1′**; honest shadow of Jelassi; backward-ref ch11/ch16 as instances; `copying-bound.png`
4. §15.4 Lyapunov diagnostics — **P2′** on constructed LTV/DPLR; degeneracy hypothesis; ch02 engine; `lyapunov-validation.png`
5. §15.5 Regime detection and effective state size — **P3′**; two-block construction; two-route cross-check; S4D-resembling example; `regime-separation.png`
6. §15.6 The mechanistic verdict — **G&A** cited remark (Bick–Xing–Gu); ch14 hybrid tie-in; the assembled toolkit; the scope line (B / Ch 17)
7. §15.7 What's next → Ch 17 (pilots integrate; these become B's instruments); back to ch13
8. §15.8 Exercises (3 short inline + 3 long) → §15.9 Full solutions → §15.10 Companion code

## 8. Companion plan

- **JAX** (`companions/ch15/jax/`):
  - `copying_bound.py` — `state_capacity_bits(d,b)`, `min_lossless_state_bits(n,vocab)`,
    `recall_cliff_load(d)`; imports `companions.ch16.jax.mqar`; figure `copying-bound.png`.
  - `lyapunov_diagnostics.py` — `constructed_lti_jacobians`, `constructed_ltv_jacobians` (via
    ch09 `discretize_selective`), `dplr_jacobians` (via ch13), S4D-resembling builder;
    `lyapunov_top` (wraps ch02 `qr_lyapunov`), `closed_form_log_growth`, `effective_state_size`;
    figures `lyapunov-validation.png`, `regime-separation.png`.
- **torch** (`companions/ch15/torch/`): `lyapunov_diagnostics.py` parity mirror (compute+parity
  only, no figures); JAX↔torch `< 1e-9` float64. **No `copying_bound` port** (ceremony).
- **Julia** (`companions/ch15/julia/`): `lyapunov_crosscheck.jl` stdlib-only QR/`eigen` Benettin
  on the same constructed transitions; `module Ch15Lyapunov` wrapper; JAX↔Julia spectra `< 1e-9`.
  Earns its place: Lyapunov spectra are a canonical NA object and the chapter's headline diagnostic.
- **Tests** (STYLE.md §8 bar): exact/integer identities `rtol=0`; P2′ vs closed form `atol<1e-6`
  on non-degenerate systems; `Σλ` vs `log|det|` tight; **degenerate block** elementwise loose +
  mean tight (the explicit caveat pin); P3′ two-route label agreement `rtol=0`; participation
  ratio `→ r`; slot cliff at `N=d` (reuse ch16 pins); every caption number pinned; `--import-mode=importlib`.

## 9. Gate items + gotchas

- Run `claim-skeptic` + `chapter-auditor` (+ `companion-verifier` + `prose-pedagogy-reviewer`)
  before advancing `status:`. **Claim-skeptic pressure points:** P1′/P2′ non-duplication
  (higher-altitude framing + "instances" backward-refs); P2′ degeneracy hypothesis stated not
  hidden; TC⁰/illusion phrased "of [authors], [year]"; RWKV-7 "all regular languages" verbatim;
  no figure secretly trains/probes a fitted model; P3′ non-circular (two-route on ground-truth-known split).
- Companions-first (prose cites *measured* numbers from companion stdout).
- Commit-time gitleaks (§4); explicit `git push -u origin feat/ch15-counter-evidence`; torch
  tests via Makefile targets (no `PYTHONPATH=.`); Julia stdlib-only + juliaup lock recovery.
- **Every inline `$...$` span on ONE physical line** (`npm run build` is the only MDX compiler;
  validate does not compile MDX); no `+`/`-`/`*`-leading continuation inside an open `$`-span.
- Never hard-code theorem numbers (`<XRef>` self-refs; labels.json counts propositions; no XRef
  inside Figure captions).
- After shipping: `CURRENT_WORK.md` (next = Ch 17), roadmap memory, `docs/DASHBOARD.md`,
  `docs/STATUS.md`, playbook memory (the duplication-trap → higher-altitude lesson).
