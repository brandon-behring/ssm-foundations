# Current Work — ssm-foundations

One-screen resume context (the `sessions.md` §1 pattern). Update on context switch / session end.
For the durable index, see the Claude Code project-memory `MEMORY.md`.

**Right now:** **Polish campaign — Part 3 (beyond-ssm, ch 11–13) shipping** as PR `polish/part3-beyond-ssm`.
Deep per-part refinement: **6 genuine fixes** (the fan-out surfaced 8 worth_it; **2 rejected after verification** —
the verify-before-fix discipline: running the ch13 overflow code showed the "~710" caption is *correct* (the
reviewer's "→730" was a miscalc that would have introduced an error), and a ch11 §11.10 "missing PyTorch
subsection" would have *broken* Part-3's consistent folded §X.10 format). Shipped: ch11 `gated_masked`
shape-validation guards (jax+torch, fail-loud parity with siblings); ch12 Fig 12.1 `Theorem 11.4`→**`Proposition
11.4`** (matches labels.json + the body XRef); ch13 trigger gloss "data stream"→"fixed layer schedule" (matches
ch14) + `\seqlen` macro trim + 2 dead figure constants removed. Gate green (build + `make check-local-torch` +
ruff, E402 still 0); re-sweep all-clean. **Touch-only formatting.**

Before this, **Part 2 (ssm-core, ch 7–10) shipped + deployed + live-verified** (PR #43, merge `066d85b`): 13
fixes — ch10's 8 raw inline-code theorem IDs → `<XRef>`, §4.7→§4.5, §10.10 PyTorch block; ch07/08/09 factual +
doc + macro nits.

Before this, **Part 1 (foundations, ch 1–6) shipped + deployed + live-verified** (PR #42, merge `00c0f9b`):
~30 fixes (wrong cross-ref/theorem numbers, consistency, the STYLE §8 torch-listing gap, prose, code/bib);
the re-sweep there caught + fixed 1 self-introduced ch06 error.

Before this, the **post-M7-staging cleanup PR** (`chore/repo-cleanup-banner-bib-astro`) shipped: closed stale
audit-umbrella **#1**, corrected README:5's banner claim, de-anonymized the F21 bib entry (Halloran et al.,
TMLR 2025), bumped **astro `^6.1.7`→`^6.4.6`**.

Before this, **M7 stage 1 (staging the pilot-results fold-in) shipped** (PR #40 + post_transformers #55):
`docs/m7-pilot-integration-plan.md` = the fold-in catalog (ch17 §17.2→§17.2b / §17.3→§17.3b / §17.5
rewrite) + the C1/B data contracts (idealized→trained-model inputs) + the flip mechanism. M7 itself (fold
C1/B trained-model results into Ch 14–17, flip alpha→beta) stays readiness-gated on the pilots in
post_transformers. Stages 1 (#32–#36), 2 (#37–#38), 3/R38 (#39) before it are all done + deployed.

**Why:** the six-chapter campaign (approved 2026-06-10, order **12 → 14 → 16 → 13 → 15 → 17**) is
**COMPLETE — 6/6.** The 17-chapter book is content-complete; cadence proven **11×** (Ch 7–17).

**Next step:** **Polish campaign — Parts 4–5 + capstone.** Parts 1–3 (foundations + ssm-core + beyond-ssm) shipped (above);
remaining batches: **14–16 (integration) / 17 (synthesis)**, then a
cross-book consistency capstone. Per-batch method (proven on Parts 1–3): review-fan-out Workflow (prose-pedagogy
+ chapter-auditor + code-idiom per chapter) → triage to genuine `worth_it` findings (reject churn) → fix +
**verify-every-edit re-sweep** (claim-skeptic + chapter-auditor on the diff) → one PR through the gate. Scope
confirmed 2026-06-14: **deep per-part refinement** (Workflow opt-in) + **touch-only formatting**. Durable
definition: the `polish-campaign` project memory.

Separately, **M7 completion** stays readiness-gated on the C1/B pilots producing trained-model results in
`post_transformers` (turnkey via `docs/m7-pilot-integration-plan.md` + post_transformers #55). Otherwise
only low-priority Track-C housekeeping (katex 0.17 bump, F37 dep pins, F22/F25 bib — F10-astro + F21 now
done). **Hold `alpha` until M7.**

Deferred-with-notes in `audits/2026-06-13_post-m6_recheck.md` (accepted, not bugs): R23 (ch08
resolvent — a self-flagged non-implemented sketch), R24 (ch09 residual magnitudes — prose already says
"measured … pinned below 1e-12"). **R38 resolved 2026-06-14** (ch10 Mamba-3 fact-check vs the local
`2603.15569.pdf`: λ=σ(u_t) + RoPE-on-B,C verified correct; MIMO reframed as also a quality lever).
Pilots execute in `post_transformers` (M6 = issue #54).

**Context when I return:**
- Per-chapter cadence (for reference / future books): brief → `/exploring-options` (4 questions) →
  companions-first → prose → wire-up → all four review subagents → one PR (doc-sync rides IN it) →
  merge=deploy → memory. Gates: `make check-local-torch` + `npm run build` (the only MDX compiler).
- **MDX/gotcha catalog (the full campaign's lessons):** every inline `$...$` span on ONE physical
  line (a `-`/`+`/`*`-leading wrapped continuation breaks acorn — ch13); an unquoted frontmatter
  `description:` containing `: ` breaks the build (YAML map — ch15); **the validator greps for
  `<Theorem` even inside `{/* */}` macros comments → never write the literal `<Theorem>` in a
  comment, use "Theorem" without brackets (ch17)**; matplotlib mathtext ≠ KaTeX (`\*` invalid, use
  `^*`); never hard-code Theorem numbers (XRef self-refs; no XRef inside Figure captions).
- **Synthesis-chapter pattern (ch17):** integration companions COMPOSE existing instruments into a
  NEW measured signature (reductions-to-components are consistency checks, not the headline);
  status:implemented needs prose+exercises+companions, NOT theorems (0 is fine); STYLE §13
  positional accommodation for <6 content sections (Ch 5 precedent); verify numerical behavior
  empirically before stating it (the secular-vs-endpoint metric distinction).
- Post-ship checklist (drift guard): a chapter PR updates CLAUDE.md status lines, README (banner +
  row), `docs/DASHBOARD.md` (row + verified + trust notes), regen `docs/STATUS.md`, refresh this file.
