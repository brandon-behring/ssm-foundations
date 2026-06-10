# Current Work — ssm-foundations

One-screen resume context (the `sessions.md` §1 pattern). Update on context switch / session end.
For the durable index, see the Claude Code project-memory `MEMORY.md`.

**Right now:** Ch 1–12 `implemented`, shipped, and deployed. Ch 12 (delta-rule lineage:
DeltaNet, Gated DeltaNet, Kimi Linear) authored vertical 2026-06-10 — five JAX modules (week12
port + greenfield gated_delta), torch parity mirrors, Julia stability companion (stdlib), four
figures, all four review subagents run and their findings fixed pre-ship. Notable correction
landed with it: Longhorn is **backward Euler** (the prox step evaluates the gradient at the
endpoint), not implicit midpoint — ch11 §11.7's hand-off line amended in the same PR.

**Why:** Foundations (1–6) + the SSM spine (7–10) + beyond-SSM (11 linear attention/Hyena,
12 delta-rule lineage) are done. The six-chapter campaign plan (approved 2026-06-10) continues
in the refined order **12 → 14 → 16 → 13 → 15 → 17**.

**Next step:** **Ch 14 — *Hybrid architectures and gating mechanisms*** (B-pilot anchor;
largest greenfield risk — `week16/hybrid_mad.py` is a stub; lean on the
`hybrid_production_2026/`, `gating_design_space/`, `memory_hybrids/` dossiers). Start with the
playbook step-0: brief at `docs/briefs/ch14-hybrid-architectures.md` + `/exploring-options`.
Ch 14 can now cite DeltaNet/Gated DeltaNet backward via `<XRef>` (the reason 12 shipped first).

**Context when I return:**
- **B unblocks after Ch 16** (needs exactly {12, 14, 16}; 12 done). C1 remains satisfied by Ch 1–10.
- Ch 14 + Ch 16 are authored against the B pilot's two-timescale-benchmark needs (pilot policy);
  Ch 12 §12.6 planted the two-limit/singular-perturbation teaser they pick up.
- Ch 12 hand-offs to honor: Ch 13 owns RWKV-7's generalized delta rule + xLSTM matrix memory
  (promised in §12.7); Ch 14 owns layer-ratio/gate-granularity design space (promised in §12.6);
  xLSTM/RWKV-7 mentions in Ch 14 stay forward-refs to Ch 13.
- Per-chapter cadence: brief → `/exploring-options` (4 standing questions) → companions-first →
  prose → wire-up → all four review subagents → one PR → merge=deploy → this doc-sync checklist.
- Open quality items (non-blocking): retroactive `claim-skeptic` sweep over Ch 1–10 (DASHBOARD
  trust note) — slot before any beta promotion; issue #14 (landing subtitle) blocked on upstream
  #135; upstream #126 (auto-numbered headings) — re-bump when it ships; issues #1 (standards
  hardening, P2) / #4 (ch04 Julia in default gate, P3).
- Post-ship checklist (drift guard): a chapter-ship PR must update CLAUDE.md status lines,
  README (banner + table row), `docs/DASHBOARD.md` (row + verified + trust notes), regenerate
  `docs/STATUS.md`, and refresh this file.
