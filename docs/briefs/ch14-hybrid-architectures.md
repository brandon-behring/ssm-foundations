# Chapter research brief — Ch 14: Hybrid architectures and gating mechanisms

> Filled 2026-06-10 from the Ch 14 kickoff recon (two Explore agents over
> ssm-foundations + post_transformers: forward-promise grep, `labels.json`,
> `bibliography.bib`, dossiers `hybrid_production_2026/` + `gating_design_space/` +
> `memory_hybrids/`, B kickoff doc). §6 decisions resolved the same day via
> `/exploring-options` (4 questions, all recommendations accepted).

- **Chapter / slug:** ch 14 — `ch14-hybrid-architectures.mdx`
- **Part / status target:** `integration` → `implemented` (first chapter of the integration part)
- **One-line scope:** Hybrids as *two-timescale* architectures — attention is exact on the fast,
  token-local process (boundary layer) while a decaying state tracks the slow latent (slow
  manifold); composition (sequential/parallel/interleaved ratio) and gate granularity
  (scalar/vector/block) are the design variables; the production lineup (as of May 2026) and
  MAD-style synthetic probes are the instances and the measurement seed.
- **Pilot tie-in:** **B (two-timescale) — anchor chapter.** B's kickoff doc supplies the task
  spec (slow Markov regime × fast regime-conditioned bigram) and the matched-asymptotics
  framing; the companion's HMM task + idealized-predictor analysis IS B's seed. Ch 16 builds
  the full 5-axis protocol; B unblocks book-side after {12, 14, 16}.

## 1. Forward-promises to redeem

| Source (file:line) | Promise made | How this chapter honours it |
|---|---|---|
| `ch12:396-402, 425-429` | Kimi's "layer-ratio and gate-granularity choices belong to Chapter 14's hybrid design space"; "the layer-ratio and gate-granularity decisions we deferred become the design variables" | §14.3 (composition/ratio) + §14.4 (granularity) make exactly these the chapter's design axes |
| `ch12:411-413` | "Pilot B's two-timescale benchmarks live exactly in that bracket — attention as the fast boundary layer, the decaying state as the slow manifold — and Chapters 14 and 16 build the architectures and the measurement protocol" | §14.2 derives the lens (Tikhonov-style theorem); §14.6 seeds the benchmark; protocol → Ch 16 |
| `ch11:332-333` | "Chapter 14 mixes these layers with attention into the production hybrids" | §14.5 production lineup, with the Ch 11/12 layer families as the mixed-in components |
| `ch12:172` | write-rule gradient dynamics "Chapter 14's hybrids will pick up" | §14.1 frames hybrids as combining the *write-limited* fast path with the *capacity-limited* slow path |
| `ch01:48` | hybrids listed among the book's discrete realizations of continuous systems | §14.2's fast-slow ODE grounds the hybrid in the continuous view |

## 2. Backward-reference anchors (→ `<XRef>` targets)

All confirmed live in `labels.json`:

| Target id | What it is | Where this chapter leans on it |
|---|---|---|
| `ch11:linattn-capacity` | Thm 11.4 (linear-attention capacity wall) | §14.1 — why pure fast-path models hit a wall |
| `ch12:stability-dichotomy` | Thm 12.4 (explicit/implicit stability) | §14.1 — the write-rule thread; §14.4 gate stability remark |
| `ch09:ssd-duality` | Thm 9.5 (SSM ≡ masked linear attention) | §14.2 — the two limits are duals, not strangers |
| `ch11:gla-ltv-duality` | Thm 11.2 (GLA decay masks) | §14.4 — vector-gate vocabulary |
| `ch12:wy-representation` | Thm 12.5 (chunkwise WY form) | §14.3 — why interleaving preserves chunkwise efficiency |
| `ch06:be-a-stable` | Thm 6.1 (implicit-method stability) | §14.2 margin/remark — stiffness language for fast-slow |

## 3. Predecessor reuse

- **High reuse (port):** none — `week16/hybrid_mad.py` is a 4-line TODO stub. Companions are
  **greenfield** (the campaign's largest greenfield risk, planned for).
- **Greenfield (author from paper math + B kickoff):** minimal hybrid block (sliding-window
  attention + gated-decay SSM; sequential/parallel/interleaved compositions; scalar/vector
  gates); two-timescale HMM task + exact filter baselines (B kickoff
  `notes/research_kickoff_b_two_timescale_benchmarks.md` §2/§3/§5).
- **Reference-only (anchor, don't copy):**
  - `hybrid_production_2026/dossier/01_production_hybrids.md` — 6 production models with
    ratios/gating/claims; `02_cross_vendor_analysis.md` — Lee 2025 sequential-vs-parallel.
  - `gating_design_space/dossier/02_cross_reference_table.md` — the granularity/schedule/
    trigger taxonomy; `01_observability_triggered_gating.md` — AMOR + Basu frontier.
  - `memory_hybrids/dossier/03_hybrid_attention_ssm.md` + `06_production_hybrid_lm.md`.
  - Third-party reference code (consult, never copy): `experiments/refs/recurrentgemma/jax/griffin.py`,
    `experiments/refs/mad-lab/`, `experiments/refs/zoology/` (MQAR + hybrid mixers).

## 4. Bibliography adds

- **Present:** `kimiteam2025kimi` (2510.26692), `yang2024deltanet`, `yang2025gateddeltanet`,
  `dao2024mamba2`, `gu2024mamba`, `poli2023hyena`, `arora2024zoology`, `yang2024gla` — the
  Ch 9/11/12 stack. **Zero hybrid entries exist.**
- **To add (~12; verify authors/titles against arXiv at add time):**
  - `de2024griffin` — De et al. 2024, arXiv:2402.19427 (Griffin)
  - `lieber2024jamba` — Lieber et al. 2024, arXiv:2403.19887 (Jamba)
  - `ren2024samba` — Ren et al. 2024, arXiv:2406.07522 (Samba)
  - SambaY / Phi-4-mini-flash — Ren et al. 2025, arXiv:2507.06607 (GMU; key per title firstword)
  - Hunyuan TurboS — Tencent 2025, arXiv:2505.15431 (AMF/MF macro-blocks)
  - Nemotron-H — Blakeman et al./NVIDIA 2025, arXiv:2504.03624 (~8% attention)
  - Bamba — IBM et al. 2024, `@misc` HF blog (no arXiv; open-data hybrid)
  - `poli2024mechanistic` — Poli et al. 2024, arXiv:2403.17844 (MAD)
  - Based — Arora et al. 2024, arXiv:2402.18668 (recall-throughput frontier; firstword per title)
  - `lee2025understanding` — Lee et al. 2025, arXiv:2510.26912 (sequential vs parallel)
  - AMOR — Zheng et al. 2026, arXiv:2602.13215 (entropy-triggered gating)
  - Basu 2026, arXiv:2603.20997 (content-based routing needs pairwise computation)
  - *Optional during drafting:* Jamba-1.5 (2408.12570), Hymba (2411.13676, parallel-head row)
- **Defer to a later chapter:** xLSTM (`beck2024xlstm`), Titans (2501.00663), RWKV-7 → Ch 13;
  LRA/RULER/MQAR-protocol keys → Ch 16.
- **gitleaks watch:** ~12 new bibkeys + arXiv ids → expect entropy hits; append printed
  `…:generic-api-key:LINE` fingerprints to `.gitleaksignore` at commit time (Ch 12 precedent).

## 5. Scope tensions / boundaries

- **Ch 13 hand-off (strict):** xLSTM exponential gates, Titans test-time memory, RWKV-7 rows
  appear in the §14.4 taxonomy table as *forward-refs only* — named, one-line, no derivation.
- **Ch 16 hand-off (strict):** MAD = philosophy summary + our one seeded task here; the full
  4-tier/5-axis evaluation protocol, LRA/RULER/MQAR survey → Ch 16. The two-timescale *task*
  is defined here (B seed); the two-timescale *protocol* (per-layer probing, disentanglement
  axis) → Ch 16.
- **Lens overclaim risk (resolved by Q3):** the correspondence attention=boundary-layer /
  SSM=slow-manifold / hybrid=matched-asymptotics is an **interpretive map**, never a proven
  property of trained networks. The theorem is about linear fast-slow *systems*; Lee 2025 is a
  *consistency check*, not a confirmation. Claim-skeptic tripwire — word it that way from the
  first draft.
- **Production volatility:** every lineup claim date-stamped "as of May 2026"; summary depth
  only (Ch 12's Kimi precedent); Bamba cited via HF blog `@misc` (no arXiv exists).
- **MoE-on-hybrid:** margin-note depth only (frontier pointer, not content).

## 6. Decisions for `/exploring-options` — RESOLVED 2026-06-10 (4 questions, all recommendations accepted)

1. **Companion languages → JAX + torch only.** No Julia: architecture-chapter precedent
   (Ch 8/9); the numerical core (HMM filter, masked attention) is linear algebra, not the NA
   stability theory that earned Julia in Ch 10/12. B's downstream code is JAX.
2. **Section allocation → lens-first arc:** two-timescale lens → composition space → gating
   space → production lineup → MAD + benchmark seed (sketch in §7 confirmed).
3. **Lens explicitness → theorem + flagged interpretation:** prove Tikhonov-style tracking for
   linear fast-slow systems; state the architecture correspondence as interpretation with the
   Lee 2025 consistency check; never claim trained hybrids implement matching conditions.
4. **Lineup depth → full lineup, summary depth:** 7-model cited table (Griffin, Jamba,
   Samba/SambaY, Bamba, Nemotron-H, Hunyuan TurboS, Kimi), each one architecture-summary deep,
   date-stamped; AMOR/Basu included in the gating taxonomy. ~12 bib adds accepted.

## 7. Likely chapter shape (sketch — confirmed against §6 answers)

1. §14.1 Two limits of sequence modeling (recall-throughput frontier; `ch11:linattn-capacity` +
   `ch12:stability-dichotomy` entry; the fast/slow problem statement)
2. §14.2 The two-timescale lens (linear fast-slow system; Tikhonov-style tracking theorem;
   boundary-layer/slow-manifold interpretation, flagged; matched-asymptotics map; Lee 2025
   consistency check)
3. §14.3 The composition design space (sequential / parallel / interleaved; layer ratio; cost
   accounting proposition — KV cache + state vs (r, w, d))
4. §14.4 The gating design space (granularity: scalar → vector → block/GMU, with exact-reduction
   proposition; schedule + trigger axes; AMOR frontier; Basu pairwise-necessity caveat;
   Ch 13 forward-refs)
5. §14.5 The production lineup, as of May 2026 (7-row cited table: ratio | gating | headline;
   sequential-vs-parallel finding; MoE margin note)
6. §14.6 MAD and the two-timescale benchmark seed (synthetic-probes philosophy; the HMM task;
   measured figure numbers; what transfers to Ch 16)
7. §14.7 What's next → §14.8 Exercises (3 short incl. code + 3 theory) → §14.9 Full solutions →
   §14.10 Companion code

## 8. Companion plan

- **JAX** (`companions/ch14/jax/`):
  - `hybrid_block.py` — sliding-window causal attention (≡ full attention when w ≥ L);
    gated-decay diagonal SSM layer (simplified-GLA style); `sequential` / `parallel_gated`
    (scalar + vector g, exact reductions at g ∈ {0,1}) / `interleave(ratio)`; `cost_accounting`
    (KV cache + state sizes, asserted against real array shapes). Figure: design-space map,
    cost-frontier vs ratio (production points placed).
  - `two_timescale.py` — HMM task generator (K regimes, switch prob ε, per-regime bigram);
    exact forward filter (Bayes-optimal); `window_filter(w)` (uniform-prior restart — the
    attention idealization); `decay_filter(λ)` (forgetting on sufficient stats — the state
    idealization); per-step cross-entropy, pinned PRNG keys. Figures: excess-CE vs ε;
    error vs w crossover.
- **torch** (`companions/ch14/torch/`): eager mirrors of both modules, parity vs JAX `<1e-9`,
  compute-only (no figures), buffers not Parameters.
- **Julia:** none (Q1).
- **Tests** (STYLE §8 bar): exact identities `<1e-12` `assert_allclose(rtol=0)` — window ≡ full
  at w ≥ L (attention AND filter); gate reductions g ∈ {0,1}; uniform-bigram ⇒ all predictors
  CE = log V; ε = 0 ⇒ posterior concentrates; normalization; cost formulas == traced shapes;
  optimality ordering (full filter ≤ window/decay, seeded); independent oracle for the filter
  (brute-force Bayes posterior by enumeration on tiny instances — claim-skeptic lesson);
  caption numbers asserted; `--import-mode=importlib`.

## 9. Gate items + gotchas

- Companions FIRST; captions quote measured stdout numbers (F19) and credit producer script +
  pinning test, with attributions matching committed assertions *exactly*.
- Figure data in sensible stable regimes; no tautological measurements (filter-vs-filter
  comparisons backed by the enumeration oracle).
- `<XRef>` self-refs for this chapter's own theorems (labels.json counts propositions — never
  hard-code "Theorem 14.N"); no `<XRef>` inside `<Figure caption="…">` props.
- Exercise headings: `### Exercise 14.M (short|short, code|theory)` / `### Solution to
  Exercise 14.M` (generate-status.mjs counts these).
- Makefile: append `companions/ch14/torch` to `companion-torch-tests`; **verify the Edit
  applied**; julia loop untouched.
- Local gates before push: `make check-local-torch` AND `npm run build` (validate does not
  compile MDX — the Ch 12 deploy-failure lesson). Watch `- `-leading wrapped math lines.
- Branch `feat/ch14-hybrid-architectures`; commit-time gitleaks (§4); explicit `git push -u`;
  PR → CI → merge-commit → deploy → live spot-check.
- After shipping: CLAUDE.md, README, `docs/DASHBOARD.md`, regen `docs/STATUS.md`,
  `CURRENT_WORK.md` (next = Ch 16), roadmap memory (M2: B's primary surface live).
