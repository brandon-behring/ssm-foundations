# Current Work — ssm-foundations

One-screen resume context (the `sessions.md` §1 pattern). Update on context switch / session end.
For the durable index, see the Claude Code project-memory `MEMORY.md`.

**Right now:** Ch 1–16 `implemented`, shipped, and deployed. Ch 15 (counter-evidence and
diagnostic tools — integration, "the prosecution's file") authored vertical 2026-06-13 —
greenfield JAX companions (`copying_bound` + `lyapunov_diagnostics`, 25 tests) + torch parity
mirror (11 tests) + a stdlib **Julia** QR-Lyapunov cross-check (12 tests), three figures, +3 bib
adds (Merrill–Petty–Sabharwal illusion-of-state; Merrill–Sabharwal TC⁰/parallelism; Bick–Xing–Gu
Gather-and-Aggregate), all four review subagents run and findings fixed pre-ship. **Impossibility-first**
arc: a *cited-not-proven* ceiling (TC⁰, the illusion of state, the copying separation) plus three
*proven* propositions — (1) an **architecture-agnostic capacity bound** (pigeonhole; ch11's rank
wall and ch16's slot model are *instances* it backward-refs, not a re-derivation — the
duplication-trap → higher-altitude lesson), (2) the **Lyapunov estimator** (recovery O(1/T); the
divergence identity Σλ=⟨log|det J|⟩ exact; the resolution limit is about *coupling/non-normality*,
not degeneracy — S4D-Lin's degenerate-but-decoupled spectrum recovers exactly, ch2's non-normal
ring scatters), (3) **regime separation via effective state size** with a two-route
(algebraic↔dynamical) anti-circularity cross-check. All diagnostics validated on *constructed*
systems; trained-model probing is pilot B's program. No pilot milestone rides on Ch 15 (C1 closed
at Ch 10, B at Ch 16) — but it builds the instruments B uses.

**Why:** The six-chapter campaign (approved 2026-06-10) ran in the refined order
**12 → 14 → 16 → 13 → 15 → 17**. 12, 14, 16, 13, 15 are done — **one chapter left**.

**Next step:** **Ch 17 — *Niche-pilot integration*** (synthesis, the campaign's last chapter).
Start with the playbook step-0: brief at `docs/briefs/ch17-<slug>.md` + `/exploring-options`.
**Step-0 flag:** the standard 6-content-section skeleton may not fit a synthesis chapter — decide a
sanctioned STYLE §13 deviation *deliberately* at kickoff (Ch 1 and Ch 5 are the precedents). Ch 17
integrates the two research pilots (C1 symplectic integrators; B two-timescale benchmarks): it takes
Ch 15's diagnostic toolkit and Ch 16's benchmark protocol and points them at the pilots, and weighs
Ch 15's counter-evidence against the fourteen chapters of construction. Likely low new-companion
weight (it composes existing instruments); confirm at kickoff. C1 draws on Ch 1–3 + 6 + 10; B on
Ch 12 + 14 + 15 + 16.

**Context when I return:**
- Per-chapter cadence: brief → `/exploring-options` (4 standing questions) → companions-first →
  prose → wire-up → all four review subagents → one PR (doc-sync rides IN it) → merge=deploy →
  memory updates. Gates: `make check` / `make check-local-torch` + the companion suites +
  `npm run build` (the only MDX compiler — validate does NOT compile MDX).
- **MDX gotchas (both seen this campaign):** (1) keep every inline `$...$` span on ONE physical line
  — a wrap whose continuation *starts* with `-`/`+`/`*` breaks acorn (the ch13 failure). (2) an
  unquoted `description:` containing `: ` (colon-space) is parsed as a YAML mapping → "bad
  indentation" build error (the ch15 failure — rephrase to drop the colon, or quote). Also: matplotlib
  figure labels use mathtext, not KaTeX — `\*` is invalid there (use `^*`).
- Open quality items (non-blocking): retroactive `claim-skeptic` sweep over Ch 1–10 (DASHBOARD
  trust note; claim-skeptic now exercised Ch 11–16) — slot before any beta promotion; STYLE.md §8
  companion-section shape is stale vs ch14/ch16 lived practice (Track-C doc touch); issue #26
  (generate-status `--check` only validates the Verified date); issue #14 (landing subtitle) blocked
  on upstream #135; upstream #126 (auto-numbered headings) — re-bump when it ships; issues #1
  (standards hardening, P2) / #4 (ch04 Julia in default gate, P3).
- Post-ship checklist (drift guard): a chapter-ship PR must update CLAUDE.md status lines, README
  (banner + table row), `docs/DASHBOARD.md` (row + verified + trust notes), regenerate
  `docs/STATUS.md`, and refresh this file.
