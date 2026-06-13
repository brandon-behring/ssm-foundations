# Current Work — ssm-foundations

One-screen resume context (the `sessions.md` §1 pattern). Update on context switch / session end.
For the durable index, see the Claude Code project-memory `MEMORY.md`.

**Right now:** Ch 1–14, 16 `implemented`, shipped, and deployed. Ch 13 (exponential gates and
matrix memory: xLSTM, RWKV-7 — beyond-ssm) authored vertical 2026-06-13 — greenfield JAX
companions (`generalized_transition` + `xlstm`, 46 tests) + torch parity mirrors (12 tests) +
a stdlib **Julia** stabilizer module (58 tests; the genuine Julia decision this chapter, ch12
precedent), three figures, +3 bib adds (xLSTM/Beck, RWKV-7/Peng, Titans/Behrouz), all four
review subagents run and their findings fixed pre-ship (claim-skeptic verified all three
propositions against the authoritative NXAI xLSTM reference implementation). Unification-first
arc around the **generalized diagonal-plus-rank-one transition** $A_t = \mathrm{Diag}(w_t) -
c_t a_t a_t^\top$: three propositions (generalized-transition spectrum; RWKV-7's *exact*
reduction to ch12's gated DeltaNet; the exponential-gate stabilizer as an *exact* change of
variables, not an approximation), the learned-direction decoupling (eviction without
overwrite), and Titans named as the fourth (*loss*) trigger class at summary depth (no
companion, by design). No pilot milestone rides on Ch 13 (C1 closed at Ch 10, B at Ch 16).

**Why:** The six-chapter campaign (approved 2026-06-10) continues in the refined order
**12 → 14 → 16 → 13 → 15 → 17**. 12, 14, 16, 13 are done.

**Next step:** **Ch 15 — *Counter-evidence and diagnostic tools*** (integration). Start with the
playbook step-0: brief at `docs/briefs/ch15-<slug>.md` + `/exploring-options`. It owns what
Ch 13 and Ch 16 forward-referenced: the copying / $\mathsf{TC}^0$ **impossibility theory** (the
ceiling RWKV-7's "recognizes all regular languages" claim brushes against — ch13 §13.3/§13.5,
ch16 §16.2) and the stability **diagnostics on trained models** (Lyapunov exponents, regime
detection / effective state size) that probe the matrix-memory stability questions Ch 13 raised
at the architecture level. Predecessor `experiments/jax/week18/lyapunov_ssm.py` is a likely
partial-reuse stub (verify at kickoff). The Julia-for-stability question is live again here
(Lyapunov spectra are a natural NA cross-check).

**Context when I return:**
- Then Ch 17 (synthesis — step-0 flag: standard skeleton may not fit; decide a sanctioned
  STYLE §13 deviation deliberately).
- Per-chapter cadence: brief → `/exploring-options` (4 standing questions) → companions-first →
  prose → wire-up → all four review subagents → one PR (doc-sync rides IN it) → merge=deploy →
  memory updates. Gates: `make check` (CI) + the companion suites + `npm run build` (validate
  doesn't compile MDX; keep every inline `$...$` span off any line that *starts* with `-`/`+` —
  a block-marker-leading wrap breaks acorn, the one ch13 build failure, fixed by joining the span).
- Open quality items (non-blocking): retroactive `claim-skeptic` sweep over Ch 1–10 (DASHBOARD
  trust note) — slot before any beta promotion; STYLE.md §8 companion-section shape is stale vs
  ch14/ch16 lived practice (Track-C doc touch); issue #26 (generate-status `--check` only
  validates the Verified date); issue #14 (landing subtitle) blocked on upstream #135; upstream
  #126 (auto-numbered headings) — re-bump when it ships; issues #1 (standards hardening, P2) /
  #4 (ch04 Julia in default gate, P3).
- Post-ship checklist (drift guard): a chapter-ship PR must update CLAUDE.md status lines,
  README (banner + table row), `docs/DASHBOARD.md` (row + verified + trust notes), regenerate
  `docs/STATUS.md`, and refresh this file.
