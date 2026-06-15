# Current Work — ssm-foundations

One-screen resume context (the `sessions.md` §1 pattern). Update on context switch / session end.
For the durable index, see the Claude Code project-memory `MEMORY.md`.

**Right now:** **Polish campaign — COMPLETE (2026-06-14).** Final step: the cross-book consistency capstone shipped
as PR `polish/capstone-crossbook` — a cross-cutting notation + cross-reference sweep over all 17 chapters found
**one** genuine fix: ch06 Exercise 6.4 said "Theorem 6.1" but the backward-Euler L-stability result is
**Proposition 6.1** (`ch06:be-a-stable` is a proposition; labels.json has Proposition 6.1 + Theorem 6.2, never 6.1);
the other 7 hard-coded result refs all resolve, notation + ruff clean. **Across all six PRs (#42–#46 + the capstone),
~60 genuine fixes polished every one of the 17 chapters + companions** — repo now **fully ruff-clean (0)**, every
cross-ref resolves, touch-only formatting preserved the teaching layout. The discipline paid for itself: the Part-1
re-sweep caught a self-introduced ch06 error, and **verify-before-fix rejected findings that would have introduced
errors** (notably the ch13 "~710" caption, proven correct by *running the figure code*).

Before this, **Part 5 (synthesis, ch 17) shipped + deployed + live-verified** (PR #46, merge `21b7472`): 4 fixes —
the §17.1/§17.5 capacity-bound vs TC⁰-Merrill cross-ref recast (claim-skeptic-confirmed), a macro-comment trim, and
the last ruff finding (ch17 F541).

Before this, **Part 4 (integration, ch 14–16) shipped + deployed + live-verified** (PR #45, merge `8988960`): 8
fixes — two reading-order "What's next" recasts (ch14/ch16 forward-framed already-read chapters), ch15 `n^\star`
notation + `\seqlen` trim, ch14/15/16 companion nits (incl. clearing the ch15 F401).

Before this, **Part 3 (beyond-ssm, ch 11–13) shipped + deployed + live-verified** (PR #44, merge `e7ca009`): 6
fixes; the verify-before-fix discipline **rejected 2 of 8** findings — a would-be numerical error in a ch13
caption (caught by running the figure code) + a Part-3 §X.10-consistency break.

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

**Next step:** **The polish campaign is COMPLETE** — all 17 chapters + companions polished across PR #42–#46 + the
capstone (tasks #73–#78 all done); the repeatable per-part method (review-fan-out → triage to `worth_it` → fix →
verify-every-edit re-sweep → gated PR → deploy → live check) is preserved in the `polish-campaign` project memory.
Remaining open work is externally gated or low-priority (below).

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
