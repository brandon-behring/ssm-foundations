# M7 — pilot-results integration plan (the alpha→beta milestone)

**Status:** stage 1 (staging) — written 2026-06-14. The fold-in itself is **readiness-gated** on the
C1/B pilots executing in `post_transformers`; this document makes that fold-in mechanical and defines
the data contract the pilots must satisfy. Companion cross-repo signal: a `tracked` issue in
`brandon-behring/post_transformers`.

## What M7 is

Fold the two research pilots' **trained-model** results into the integration/synthesis chapters,
replacing the current *provisional/idealized* statements, then flip the release status **alpha → beta**.

- **C1** (primary): symplectic/geometric integrators for SSMs — does a *trained* SSM mode carry enough
  near-conservative structure off the diagonal that a structure-preserving integrator beats the
  exp-trapezoidal default? Executes as the `symplectic_atlas` program.
- **B** (parallel): two-timescale benchmarks — on *trained* hybrids, does a layer's effective state
  size govern its regime-recovery (probe) and loss, as the idealized pipeline predicts? Executes as the
  `twotimescale` program.

Both ship a **documented-null fallback** (ch17 §17.6): they produce an artifact either way, so M7 folds
in "the measured signature" *or* "the documented null."

> Cross-repo sources (absolute, pinned to `main` per the CLAUDE.md convention):
> niche commitment — <https://github.com/brandon-behring/post_transformers/blob/main/notes/niche_decision_2026_05_24.md>.

---

## 1. Fold-in catalog — what M7 updates when results land

Grouped by chapter; line numbers are approximate (prose drifts — the **section anchors + quoted
phrases** are the durable handles). The bulk is in ch17; ch14–16 are mostly cross-reference + one
trained-vs-idealized comparison each.

### Ch 17 — niche-pilot integration (the synthesis; most of the work)
- **Frontmatter `description:` (≈L6)** — "…the reproducible template the pilots fill with trained-model
  data." → note that §17.2b/§17.3b now contain the filled-in data.
- **At-a-glance (≈L44–47)** — "…on *trained* checkpoints — is the pilots' forthcoming program, so the
  chapter's verdict … is deliberately provisional." → "the pilots' program is complete (§17.2b, §17.3b);
  the verdict (§17.5) is now empirical."
- **§17.1 idealized-vs-trained framing (≈L80–83)** — "running these instruments on *trained* checkpoints
  is the pilots' empirical program, in flight, not reported here. … the data is forthcoming." → announce
  the data is now in (link §17.2b/§17.3b).
- **§17.2 C1 (anchor L85; tail ≈L120–123)** — "Whether a *trained* selective SSM drifts off the diagonal
  … this idealized cell cannot answer, only frame." → **ADD §17.2b "C1 empirical atlas: trained-matrix
  results"** (the `atlas_cell()` output run on real Mamba-3 matrices; the energy-drift/band verdict;
  any null).
- **§17.3 B (anchor L125; tail ≈L154–159)** — "Every predictor here is an exact idealization … run this
  same pipeline on *trained* checkpoints." → **ADD §17.3b "B empirical disentanglement: trained-hybrid
  results"** (per-layer ESS↔probe↔CE coherence + monotonicity on trained stacks; a new figure).
- **§17.5 "What the lens earned, and its limits" (anchor L201; "provisionally yes" L203; "every number …
  idealized … forthcoming work" ≈L215)** — **MAJOR REWRITE**: keep the 3-part structure (earned / not
  earned / limits) but replace "provisionally yes … forthcoming" with the empirical verdict (C1: did
  geometric integrators find structure in trained modes? B: does ESS govern disentanglement in trained
  hybrids? — with the correlation-vs-causation caveat). Keep the TC⁰ ceiling (§15) cited-not-raised.
- **§17.6 "Where this goes" (anchor L225; ≈L229)** — confirm C1/B complete; add result/repo links;
  note the beta flip.

### Ch 14 — hybrid architectures (pilot B's seed)
- **§14.6 reference numbers** (the optimal-filter 1.9289 nats + window/decay/unigram excesses) — keep
  idealized; **add a trained-model comparison** (same reference HMM config) once B measures it.
- **The "interpretive map between idealized computations, not a theorem about trained networks" caveat
  (≈L170–188)** — update with B's verdict on whether trained hybrids actually implement the matching
  condition.
- Cross-refs ("…is Chapter 16's subject / B's program") → point to §17.3b.

### Ch 15 — counter-evidence + diagnostics (builds B's instruments)
- **"B points it at the data" deferrals (≈L243–258, L309–312)** — keep the instrument-validation prose;
  add a footnote/short result pointing to the trained-model effective-state-size + Lyapunov measurements
  (§17.5 / a new §15.x).
- **The verdict-deferred line (≈L342–344)** → "Chapter 17 now integrates the verdict with empirical
  evidence (§17.5)."

### Ch 16 — empirical methodology (B's protocol)
- **The idealized probe-signature figure (`ch16:fig:probe-signature`, ≈L475–492)** — add a **trained-model
  companion figure** showing the measured disentanglement signature side-by-side.
- The "first trained-model consumer is pilot B" / "Chapter 17 integrates the pilots" cross-refs (≈L557,
  L566–572) → point to §17.2b/§17.3b.

---

## 2. Data contracts — what the pilots must produce (drop-in for the fold-in)

The ch17 integration companions already compute the *signatures* on idealized inputs; the pilots replace
the idealized inputs with trained-model inputs of the **same shape/format**, and the templates produce
the fold-in numbers/figures unchanged.

### C1 — `companions/ch17/jax/c1_integration.py` (`atlas_cell(dt, periods)`)
| Idealized input (now) | Trained-model input (M7) | Format |
|---|---|---|
| Single imaginary mode `λ = −i` (ch10 `complex_scalar_recurrence`) | **Trained Mamba-3 transition matrices `A`** — *diagonal* (negative-eigenvalue extensions) **and** *coupled/off-diagonal* (the real question: does structure reappear off-diagonal?) | `(D,D)` float64 |
| `(q0,p0)=(1,0)`, `H0=0.5` | Checkpoint state vectors `x0` (or seeded) | `(D,)` float64 |
| Fixed `dt ∈ {0.05,0.1,0.2,0.4}` | A `dt` sweep over the architecture's operating range | list of float |
| Verlet/RK4/exact-exp on the constructed mode | Same three integrators applied to the trained `A` (exact-exp via `expm`; Verlet/RK4 steppers from ch06) | — |

**Output signature** (per `dt`, the fold-in numbers): `{E0, exact_exp_drift, exact_exp_band,
verlet_drift, verlet_band, rk4_drift, rk4_band}` — i.e. per-integrator **secular energy drift** + **band**.
Surfaces as **§17.2b** + a figure (energy error over periods; log-log drift vs `dt`). Idealized
reference values to beat/compare: exact-exp band `<1e-10`, Verlet secular `~5.8e-9`, RK4 endpoint drift
reproduces ch06 to `1.3e-11`.

### B — `companions/ch17/jax/b_integration.py` (`disentanglement_readout()`)
| Idealized input (now) | Trained-model input (M7) | Format |
|---|---|---|
| `TwoTimescaleHMM` (K=4) + closed-form filters (full/decay/unigram) | **Trained-checkpoint layer activations** `h` for the regime-carrying layer (identified via probing) | `(T,D)` float64 |
| Ground-truth HMM regime labels | **Held-out regime labels** from the same two-timescale task (probe target, *not* refit) | `(T,)` int |
| Filter transition operators (for ESS) | The regime-layer's **Jacobian / linearization** at its operating point (→ effective state size) | `(D,D)` |
| Reference tokens | Task instances with the fast (bigram) + slow (regime) structure | `(T,)` int |

**Output signature** (the coherence verdict): per predictor/layer `{ess, probe_acc, ce}` + the paired
comparison stats + **`probe_monotone_in_ess: bool`**. Surfaces as **§17.3b** + a figure (probe-acc vs
ESS scatter; CE bars). Idealized reference: full `ess≈4.0/probe≈0.839/CE≈1.93`, decay `3.66/0.69/1.98`,
unigram `1.0/0.548/2.425`; monotone in ESS; paired SE resolves the gap to ~15σ.

### Null-result fallback
If a pilot's measured effect is null (e.g., trained diagonal SSMs already exact-exp-optimal so symplectic
buys nothing; or trained hybrids show no ESS↔probe monotonicity), the fold-in reports the **documented
null** — the chapters already frame this as a legitimate outcome (ch17 §17.6), and §17.5's "what it did
*not* earn" subsection absorbs it.

---

## 3. Release-status flip mechanism (alpha → beta)

**Finding (2026-06-14 recon):** the toolkit ships `PreReleaseBanner.astro` (`state=alpha|beta|rc|locked`),
but **this book never wired it** — there is no `<PreReleaseBanner>` in `src/`, no `releaseStatus` field
in `astro.config.mjs`, and the **live site renders no banner**. So README:5's "Pre-release banner is live
site-wide" is **stale/inaccurate**. Implication: "alpha" currently lives only as README text + the
conceptual roadmap; there is no rendered banner to "flip."

**At M7 completion, the flip is therefore:**
1. Either **wire** `PreReleaseBanner state="beta"` (in a layout/landing override, if a rendered banner is
   wanted) **or** simply correct the README text status — and **fix the stale README:5 claim** either way.
2. Doc-status sweep: README:5 (`alpha`→`beta`), `CLAUDE.md` status line, `docs/DASHBOARD.md` trust note
   ("flip `alpha → beta`"), the roadmap memory.
3. Chapter `status:` frontmatter stays `implemented` throughout (beta is a book-level maturity label,
   not a per-chapter status-taxonomy change).

(Not fixed in stage 1 — recorded here as an M7-completion item; the site stays alpha/banner-less today.)

---

## 4. Execution checklist (turnkey, when results land)

1. Pull C1/B result artifacts (matching §2's contracts) from `post_transformers`.
2. Run the ch17 integration companions on the real inputs → §17.2b (C1) + §17.3b (B) numbers + figures.
3. Write §17.2b, §17.3b; **rewrite §17.5** (provisional → empirical verdict); update §17.6 links/status;
   un-hedge the ch17 frontmatter + at-a-glance + §17.1.
4. Update ch14 (trained-vs-idealized comparison + the matching-condition verdict), ch15 (trained
   diagnostics footnote/result), ch16 (trained probe-signature figure) + all cross-refs → §17.2b/§17.3b.
5. Regenerate `docs/STATUS.md`; run `companion-verifier` (new figures/numbers) + `claim-skeptic` (the
   new empirical claims) + `chapter-auditor` on ch17.
6. Flip the release status (§3); doc-status sweep.
7. Ship through the gate (`make check-local-torch` + `npm run build` → PR → CI → merge → deploy → live
   check). Update the roadmap memory: M7 complete, book at beta.

---

## References
- Ch 17 brief: `docs/briefs/ch17-niche-pilot-integration.md`.
- The provisional-verdict origin: `audits/2026-06-13_post-m6_recheck.md` (R38 + the ch17 claim-skeptic
  notes) and the roadmap memory.
- Integration companions: `companions/ch17/jax/{c1_integration,b_integration}.py`,
  `companions/ch17/julia/symplectic_crosscheck.jl`.
