# Current Work — ssm-foundations

One-screen resume context (the `sessions.md` §1 pattern). Update on context switch / session end.
For the durable index, see the Claude Code project-memory `MEMORY.md`.

**Right now:** **The book is content-complete (M6).** All **Ch 1–17 `implemented`, shipped, and
deployed.** Ch 17 (niche-pilot integration — synthesis, the crown-jewel/last chapter) authored
vertical 2026-06-13: two JAX **integration companions** that *compose* shipped instruments —
`c1_integration` (the C1 symplectic atlas cell: ch6 Verlet/RK4 × ch10 complex-mode SSM) and
`b_integration` (the B disentanglement pipeline: ch14 HMM → ch16 protocol → ch15 effective state
size) — plus a stdlib **Julia** C1 symplectic energy cross-check (no torch by design — compositions
introduce no new kernel); two figures; **+0 bib** (all anchors present); all four review subagents
run and findings fixed. Two NEW integrated signatures (the Ch-15 duplication-trap guard): **C1** —
on a conservative SSM mode the diagonal exact-exponential dominates both Verlet and RK4, so the
symplectic advantage bites only *off-diagonal* (the pilot's real question); **B** — effective state
size ↔ regime-probe accuracy ↔ cross-entropy cohere monotonically across the predictor family.
**Honest boundary**: every demo is on *idealized* systems; trained-model results are the pilots'
forthcoming program (forward-ref'd to post_transformers). PR #31-or-next, merge = M6.

**Why:** the six-chapter campaign (approved 2026-06-10, order **12 → 14 → 16 → 13 → 15 → 17**) is
**COMPLETE — 6/6.** The 17-chapter book is content-complete; cadence proven **11×** (Ch 7–17).

**Next step:** **No next chapter.** The work shifts to the **post-M6 beta gate** (no longer
dependency-blocked — elect when ready):
- Retroactive `claim-skeptic` sweep over **Ch 1–10** (the forward gate only ran Ch 11–17) — a
  Workflow fan-out candidate, needs explicit opt-in.
- `citation-link-auditor` repo-wide sweep (bibkey hygiene + `<Cite>` resolution + cross-repo URL
  freshness) — bib is 67 entries, last +3 at Ch 15.
- STYLE.md §8 companion-section shape refresh (stale vs ch14/16/17 lived practice).
- Toolkit re-bumps if upstream `book-scaffold-astro` #126 (auto-numbered headings) / #135 ship;
  issues #26 (generate-status `--check` only checks the date), #14 (landing subtitle), #1 / #4.
- The **pilots execute empirically** in `post_transformers` (C1 symplectic_atlas, B twotimescale);
  Ch 17 shipped their integrated templates. Surfaced to post_transformers as the M6 notification.

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
