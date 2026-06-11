# Current Work — ssm-foundations

One-screen resume context (the `sessions.md` §1 pattern). Update on context switch / session end.
For the durable index, see the Claude Code project-memory `MEMORY.md`.

**Right now:** Ch 1–12, 14, 16 `implemented`, shipped, and deployed. Ch 16 (empirical
methodology: benchmark protocols and evaluation — the second **B-pilot anchor**) authored
vertical 2026-06-11 — greenfield JAX companions (`mqar` + `protocol`, 89 tests) + torch parity
mirrors (9 tests), four figures, +7 bib adds (LRA, RULER, induction heads, copying, Seif,
LongBench, SCROLLS), all four review subagents run and their findings fixed pre-ship. The
chapter ships the protocol around Ch 14's task: four propositions (discriminative regime,
paired comparison, selection inflation, probe recoverability), the deferred *composite*
predictor (two exact identities: uniform ≡ ch14 window; matched λ* ≡ full filter at every w),
the probe-signature method (B's disentanglement axis on idealized states), the distractor
rule for length stress (neutral padding is provably inert — pad with content, not blanks),
L90/AUC metrics, and the 4-tier evaluation stack. **MILESTONE M3: pilot B's book-side
prerequisites (Ch 12 + 14 + 16) are complete** — surfaced to post_transformers at ship time.

**Why:** The six-chapter campaign (approved 2026-06-10) continues in the refined order
**12 → 14 → 16 → 13 → 15 → 17**. 12, 14, 16 are done; B is unblocked book-side.

**Next step:** **Ch 13 — *Exponential gates and matrix memory: xLSTM, RWKV-7*** (beyond-ssm;
← Ch 12; MED reuse — predecessor week14/15 are TODO stubs, dossiers only; bib +~2 xLSTM/RWKV).
Start with the playbook step-0: brief at `docs/briefs/ch13-xlstm-rwkv.md` + `/exploring-options`.
Ch 12/14 hand it: the gate-interior thread (exponential gating + stabilizer states; the
generalized delta rule extending ch12's lineage to matrix memories with their own stability
questions — ch14 §14.4/§14.7 name xLSTM/RWKV-7/Titans as Ch 13 forward-refs).

**Context when I return:**
- Then Ch 15 (counter-evidence — owns the copying/TC⁰ impossibility theory ch16 §16.2 forward-
  referenced; `week18/lyapunov_ssm.py` stub partial reuse), then Ch 17 (synthesis — step-0
  flag: standard skeleton may not fit; decide a sanctioned STYLE §13 deviation deliberately).
- Per-chapter cadence: brief → `/exploring-options` (4 standing questions) → companions-first →
  prose → wire-up → all four review subagents → one PR (doc-sync rides IN it) → merge=deploy →
  memory updates. Gates: `make check-local-torch` AND `npm run build` (validate doesn't compile
  MDX; keep every inline `$...$` span on a single line — list-item wraps starting `-`/`+`
  break acorn).
- Open quality items (non-blocking): retroactive `claim-skeptic` sweep over Ch 1–10 (DASHBOARD
  trust note) — slot before any beta promotion; STYLE.md §8 companion-section shape is two
  chapters stale vs ch14/ch16 lived practice (Track-C doc touch, auditor note 2026-06-11);
  issue #26 (generate-status `--check` only validates the Verified date); issue #14 (landing
  subtitle) blocked on upstream #135; upstream #126 (auto-numbered headings) — re-bump when it
  ships; issues #1 (standards hardening, P2) / #4 (ch04 Julia in default gate, P3).
- Post-ship checklist (drift guard): a chapter-ship PR must update CLAUDE.md status lines,
  README (banner + table row), `docs/DASHBOARD.md` (row + verified + trust notes), regenerate
  `docs/STATUS.md`, and refresh this file.
