# Chapter research brief — Ch 16: Empirical methodology

> Filled 2026-06-11 from two-agent recon (ssm-foundations + post_transformers) ahead of
> authoring. Drives the `/exploring-options` round (§6 — resolved same day, 4/4
> recommendations accepted).

- **Chapter / slug:** ch 16 — `ch16-empirical-methodology.mdx`
- **Part / status target:** integration → `implemented`
- **One-line scope:** how to *measure* sequence architectures honestly — synthetic probes,
  long-context tiers, comparison statistics, and the two-timescale protocol — organized by the
  principle that a benchmark measures only in its discriminative regime.
- **Pilot tie-in:** **B (two-timescale benchmarks) — the anchor chapter.** B's book-side
  prerequisites (Ch 12 + 14 + 16) close at this chapter's merge (**milestone M3**). B needs:
  the eval-harness skeleton, the composite predictor deferred from ch14:518, the per-layer
  probing rationale (probe-recoverability prop), the 5-axis decomposition, and the 4-tier
  protocol that situates its tier-B contribution.

## 1. Forward-promises to redeem

| Source (file:line) | Promise made | How this chapter honours it |
|---|---|---|
| `ch14:55–57` | "Chapter 16 builds the measurement protocol around it (per-layer probing, the evaluation tiers), and the book-side prerequisites for B close there" | §16.5 (probing + protocol slice) + §16.6 (tiers; closure note) |
| `ch14:417–419` | "The task below is B's layer-0 deliverable; Chapter 16 supplies the protocol around it" | §16.5 wraps the §14.6 task in the protocol |
| `ch14:518–519` | the *composite* restriction — window-$w$ filter seeded at its edge with a decayed carried prior — "is the protocol's" | `protocol.py::composite_filter_predictions`, measured in §16.5 |
| `ch14:522–525` | per-layer probing, evaluation tiers, five-axis decomposition "— that is Chapter 16's subject" | §16.5 (probing, 5-axis) + §16.6 (tiers) |
| `ch12:427–429` | "Chapter 16 then builds the evaluation methodology — including the two-timescale protocol pilot B contributes — that makes the comparisons honest" | §16.3 (comparison statistics) + §16.5 |
| `ch03:242` | "Chapter 16's empirical-methodology discussion of why some architectures can copy strings exponentially longer than others" | §16.2 copying-probe row (protocol depth; impossibility theory stays Ch 15) |

## 2. Backward-reference anchors (→ `<XRef>` targets)

All confirmed resolving in `labels.json`.

| Target id | What it is | Where this chapter leans on it |
|---|---|---|
| `ch14:matched-decay-optimal` | matched-decay optimality theorem | §16.5 (the decay baseline's meaning) |
| `ch14:slow-manifold-tracking` | Tikhonov-style tracking theorem | §16.1/§16.5 (what the lens predicts) |
| `ch14:fig:two-timescale` | the seed task's excess-CE figure | §16.5 recap |
| `ch14:decode-cost` | decode-cost proposition | §16.1 (efficiency axis) |
| `ch11:linattn-capacity` | additive-state capacity wall (√(K/d) interference) | §16.2 (Prop 1 anchors on it; no re-proof) |
| `ch12:stability-dichotomy` | delta-rule stability dichotomy | §16.1 (what honest comparison must control) |
| `ch09:ssd-duality` | selective SSM ≡ masked linear attention | §16.1 (why families are comparable at all) |
| ch03 Krylov section | copying ceiling mechanism | §16.2 copying row (inline XRef if labeled; else prose ref) |

## 3. Predecessor reuse

- **High reuse (in-book, not predecessor):** `companions/ch14/jax/two_timescale.py` (task,
  filters, CE metric — imported, not copied) and `companions/ch11/jax/mqar_recall.py`
  (fixed-weight associative-recall mechanism — reuse the mechanism shape for the
  outer-product readout).
- **Greenfield:** tokenized MQAR generator + oracle; exact-capacity idealization;
  L90/AUC metrics; harness skeleton; paired/inflation statistics; composite predictor;
  ridge probe signature.
- **Reference-only (anchor, don't copy):**
  `post_transformers/notes/benchmark_survey.md` (4-tier stack, per-benchmark rows, §I
  length-robustness metrics), `notes/research_kickoff_b_two_timescale_benchmarks.md`
  (5-axis decomposition, probing protocol, deliverables),
  `references/dossier/benchmarks/` (dataset ledger). Predecessor W3 is reading-only;
  W17/W19 code are stubs — nothing to port (PyTorch-repro angle cut per crosswalk §22).

## 4. Bibliography adds

- **Present:** `arora2024zoology` (MQAR), `poli2024mechanistic` (MAD), `arora2024simple`
  (Based), `lee2025understanding`, `gu2022s4` (Path-X context), `zheng2026amor`, `basu2026content`.
- **To add (+7, authors verified via arXiv API at add time):**
  `tay2021long` (LRA, 2011.04006); `hsieh2024ruler` (RULER, 2404.06654); `olsson2022context`
  (induction heads, 2209.11895); `jelassi2024repeat` (copying, 2402.01032); `seif2022…`
  (2205.14683 — **title/authors must be verified before keying**; B's two-timescale-learning
  anchor); `bai2024longbench` (2308.14508); `shaham2022scrolls` (2201.03533).
- **Defer:** xLSTM/RWKV-7 eval rows → Ch 13; TC⁰/G&A/diagnostics → Ch 15.
- **gitleaks watch:** new bibkeys + math-heavy lines may trip entropy >3.5 → append printed
  fingerprints to `.gitleaksignore`, never `--no-verify`.

## 5. Scope tensions / boundaries

- **Ch 15 boundary (strict):** copying *impossibility theory* (TC⁰ bounds, Jelassi's
  theoretical side), G&A mechanism, Lyapunov/ESS diagnostics → Ch 15. Ch 16 takes the copying
  *probe row* (what to measure, where it discriminates) and forward-refs Ch 15. Redeems
  ch03:242 at protocol depth.
- **Ch 17 boundary:** pilot *synthesis* (what B found, how pilots integrate) → Ch 17. Ch 16
  ships the methodology and notes only that B's book-side prerequisites close here.
- **No-training constraint:** companions are deterministic. The probing demo runs closed-form
  ridge probes on *idealized filter states* (forward posterior / composite / window / decay /
  unigram), not on trained networks — trained-hybrid probing is B's empirical program in
  post_transformers. Flagged-interpretation discipline (ch14 decision 3) carries over verbatim.
- **LRA-deprecation claim:** real but easy to overclaim. Frame as sourced observation (what
  LRA tests vs what LM-relevant capability needs; S4 solving Path-X as the inflection), cite
  tay2021long + gu2022s4; no "everyone agrees LRA is dead" language.
- **Zoology's "most predictive synthetic" claim** is *their* claim — attribute, don't adopt.
- **Cross-repo links:** 4-tier/5-axis provenance points at post_transformers via absolute
  GitHub URLs pinned to main (CLAUDE.md convention).

## 6. Decisions for `/exploring-options` — RESOLVED 2026-06-11 (4/4 recommendations accepted)

1. **Companions:** two modules (`mqar.py` + `protocol.py`), JAX + torch parity, **no Julia**
   (methodology chapter, no NA core — Ch 8/9/14 precedent).
2. **Arc:** measurement-problem-first, tiers as spine; B's protocol is the §16.5 worked
   synthesis (mirrors ch14's §14.6 placement).
3. **Theorems:** four props — discriminative regime; probe-recoverability; paired-comparison
   variance reduction; selection inflation. All provable in-page; all numbers companion-measured.
4. **Bib/coverage:** full tier table, every named benchmark keyed (+~7), one-row summary depth
   (ch14 production-table precedent).

## 7. Likely chapter shape (sketch)

1. §16.1 The measurement problem (lens obligations; validity; honest-comparison promise; tier preview)
2. §16.2 Synthetic probes — tier A (MQAR/MAD/induction/copying; **Prop: discriminative regime**; figure)
3. §16.3 The statistics of honest comparison (**Props: paired variance reduction; selection inflation**; figure)
4. §16.4 Long-context evaluation — tiers B/C (LRA lesson; RULER/NIAH; LongBench/SCROLLS; L90/AUC; figure)
5. §16.5 The two-timescale protocol (composite predictor; **Prop: probe-recoverability** + limits; probe-signature figure; 5-axis)
6. §16.6 The assembled protocol (cited 4-tier table; B book-side closure)
7. §16.7 What's next → §16.8 Exercises (3 short + 3 theory) → §16.9 Full solutions → §16.10 Companion code

## 8. Companion plan

- **JAX** (`companions/ch16/jax/`): `mqar.py` (generator, scan oracle, softmax reader,
  outer-product readout, exact-capacity idealization, accuracy sweep, L90/AUC) →
  `discriminative-regime.png`, `length-robustness.png`; `protocol.py` (score harness,
  paired/unpaired stats, max-of-k inflation, composite predictor, ridge probe signature,
  5-axis/tier data) → `probe-signature.png`, `selection-inflation.png`.
- **torch** (`companions/ch16/torch/`): `mqar.py` + `protocol.py` scoring-path mirrors +
  composite predictor; parity vs JAX in-process.
- **Julia:** none (sanctioned).
- **Tests:** ~50–60 JAX + ~6–10 torch; exact identities `<1e-12` `rtol=0`; parity `<1e-9`;
  independent oracles (scan-lookup for MQAR; enumeration already exists for the HMM via ch14);
  every caption number pinned at printed precision; `--import-mode=importlib`.

## 9. Gate items + gotchas

- Companions-first; prose cites only measured numbers.
- Every inline `$...$` span on ONE physical line (`- `/`+ ` wraps break acorn at `npm run build`);
  run the odd-$ line scan + `npm run build` before push.
- Never hard-code theorem numbers (XRef self-refs; labels.json counts props; no XRef in captions).
- Prop hypotheses must cover exactly what the companion implements (claim-skeptic, ch14 S3 lesson).
- All four review subagents pre-ship; doc-sync rides IN the PR (ch14 C1 lesson).
- torch tests via Makefile target (no PYTHONPATH); JAX mains need `PYTHONPATH=.`.
- Ship: `feat/ch16-empirical-methodology`, stage chapter artifacts only, explicit `git push -u`.
- Post-merge: roadmap memory M3, playbook lessons, `gh issue` on post_transformers (B watchlist).
