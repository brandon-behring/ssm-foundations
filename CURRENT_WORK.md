# Current Work — ssm-foundations

One-screen resume context (the `sessions.md` §1 pattern). Update on context switch / session end.
For the durable index, see the Claude Code project-memory `MEMORY.md`.

**Right now:** Ch 1–11 `implemented`, shipped, and deployed. The toolkit bump PR (#20,
`book-scaffold-astro` 4.8 → 4.16.0) is **merged and deployed** (`e5f4872`, 2026-06-05) — theorem
labels render book-wide and `<XRef>` is live (exercised in ch11). A 2026-06-09 repo audit
(`/methodology-audit`, standard depth) corrected post-ship doc drift (CLAUDE.md, DASHBOARD.md,
this file) and reformatted ch11's exercises to the STYLE §7 heading convention (`### Exercise` /
`### Solution to Exercise`), fixing the STATUS.md 0-exercise miscount.

**Why:** Foundations (1–6) + the SSM spine (7–10: HiPPO → S4/S4D/S5 → Selective → Mamba-3) + the
first beyond-SSM chapter (11, linear attention + Hyena) are done. Part IV (beyond-ssm) continues.

**Next step:** **Ch 12 — *Delta-rule lineage: DeltaNet, Gated DeltaNet, Kimi Linear*** (refined
order, 2026-06-09 — see below). Start with `/exploring-options` (chapter-authoring playbook), then
companions-first. Ch 12 is B-load-bearing (the delta-rule online-learning ODE).

**Context when I return:**
- **Refined authoring order (2026-06-09): 12 → 14 → 16 → 13 → 15 → 17** (was 14 → 16 → 12 → …).
  Rationale: B is blocked on exactly {12, 14, 16}, so any order of those three unblocks B after
  three chapters — and 12-first (a) resolves the 12-vs-14 tension the 0604 checkpoint flagged
  (Ch 12 is B-load-bearing), (b) lets Ch 14 (hybrids) cite DeltaNet/Gated DeltaNet backward via
  `<XRef>` instead of forward-promising, and (c) honors Ch 11's explicit hand-off of delta-rule
  scope to Ch 12 while the GLA/decay-mask material is fresh. Sequencing stays
  dependency/readiness-gated, not dated. C1 remains dependency-satisfied by Ch 1–10.
- Ch 14 + Ch 16 are authored against the B pilot's two-timescale-benchmark needs (pilot policy).
- Use `<XRef>` for backward object-refs in Ch 12+ (STYLE §4 XRef-forward policy, live since the
  4.16.0 bump). The book is uniform on `type=` for theorem kinds.
- Control-theory lens for Ch 12–17 = *natural touches only*; a dedicated SSM/control treatment is
  a separate guide (deferred). Run each chapter through `claim-skeptic` before advancing `status:`.
- Open quality items (non-blocking): retroactive `claim-skeptic` sweep over Ch 1–10 (DASHBOARD
  trust note) — slot before any beta promotion; issue #14 (landing subtitle still scaffold
  template) — fold into the next deploy; upstream #126 (auto-numbered headings) — re-bump when it
  ships; issues #1 (standards hardening, P2) / #4 (ch04 Julia in default gate, P3).
- Post-ship checklist (drift guard, added after the 2026-06-09 audit): a chapter-ship PR must also
  update CLAUDE.md status line, README, `docs/DASHBOARD.md`, regenerate `docs/STATUS.md`, and
  refresh this file — PR #19 caught README/STATUS but missed CLAUDE.md/DASHBOARD.
