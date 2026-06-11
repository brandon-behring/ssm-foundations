# Current Work — ssm-foundations

One-screen resume context (the `sessions.md` §1 pattern). Update on context switch / session end.
For the durable index, see the Claude Code project-memory `MEMORY.md`.

**Right now:** Ch 1–12 + 14 `implemented`, shipped, and deployed. Ch 14 (hybrid architectures
and gating mechanisms — the first integration-part chapter and the **B-pilot anchor**) authored
vertical 2026-06-10 — greenfield JAX companions (`hybrid_block` + `two_timescale`, 58 tests) +
torch parity mirrors (8 tests), four figures, ~13 bib adds (the full May-2026 production
lineup), all four review subagents run and their findings fixed pre-ship. The chapter ships
pilot B's seed artifact: the two-timescale HMM task with exact filter baselines, the
matched-decay optimality theorem (T(ε)=M(λ*) — gating is timescale matching), and the
three-timescale benchmark design lesson (w ≪ τ_id ≪ 1/ε, with the overlap dial η controlling
τ_id).

**Why:** The six-chapter campaign (approved 2026-06-10) continues in the refined order
**12 → 14 → 16 → 13 → 15 → 17**. 12 and 14 are done; **B unblocks at Ch 16** (needs exactly
{12, 14, 16}).

**Next step:** **Ch 16 — *Empirical methodology: benchmark protocols*** (B-pilot anchor;
reframe W3+W17+W19 survey→methodology; LOW code reuse, HIGH reference reuse — `benchmarks/`
dossier). Start with the playbook step-0: brief at `docs/briefs/ch16-empirical-methodology.md`
+ `/exploring-options`. Ch 14 hands it: the two-timescale *protocol* (per-layer probing,
disentanglement axis), the deferred *composite* (window + carried prior) predictor, MAD/MQAR
eval-tier context, and the B 5-axis decomposition. **Milestone on merge: B unblocks book-side**
— record in roadmap memory and surface to post_transformers (B kickoff watchlist).

**Context when I return:**
- Bib candidates for Ch 16: LRA, RULER, MQAR/zoology protocol papers, Seif 2205.14683
  (two-timescale theory); MAD already added by Ch 14 (`poli2024mechanistic`).
- Ch 14 hand-offs to honor: Ch 13 owns xLSTM/Titans/RWKV-7 (named as forward-refs in §14.4);
  Ch 15 owns counter-evidence/diagnostics; Ch 16 owns everything evaluative (§14.6 closes with
  an explicit does-not-transfer list).
- Per-chapter cadence: brief → `/exploring-options` (4 standing questions) → companions-first →
  prose → wire-up → all four review subagents → one PR → merge=deploy → this doc-sync checklist.
  Gates: `make check-local-torch` AND `npm run build` (validate doesn't compile MDX; keep every
  inline `$...$` span on a single line — list-item wraps starting `-`/`+` break acorn).
- Open quality items (non-blocking): retroactive `claim-skeptic` sweep over Ch 1–10 (DASHBOARD
  trust note) — slot before any beta promotion; generate-status `--check` only validates the
  Verified date, not table content (tracked issue filed at ship time); issue #14 (landing
  subtitle) blocked on upstream #135; upstream #126 (auto-numbered headings) — re-bump when it
  ships; issues #1 (standards hardening, P2) / #4 (ch04 Julia in default gate, P3).
- Post-ship checklist (drift guard): a chapter-ship PR must update CLAUDE.md status lines,
  README (banner + table row), `docs/DASHBOARD.md` (row + verified + trust notes), regenerate
  `docs/STATUS.md`, and refresh this file.
