# Current Work — ssm-foundations

One-screen resume context (the `sessions.md` §1 pattern). Update on context switch / session end.
For the durable index, see the Claude Code project-memory `MEMORY.md`.

**Right now:** Ch-11 **runway cleared** on branch `chore/ch11-runway` (F4 XRef adopted forward-only;
F26 importlib test mode + bar codified; F27 ch04–06 torch parity → torch now complete ch01–10;
`docs/DASHBOARD.md` + chapter-research-brief template added). PR open — **merging deploys** (ch04–06
§X.10 edits touch `src/`). Ch 1–10 `implemented`/shipped; `main` at `4dbcd92` (PR #15 merged).

**Why:** Foundations + the SSM spine (HiPPO → S4/S4D/S5 → Selective → Mamba-3) are done. Part IV
(beyond-ssm) is next, opening with Ch 11.

**Next step:** Ch 11 — *Linear attention and Hyena*. Start with `/exploring-options` (per the
chapter-authoring playbook), then companions-first. High predecessor reuse: `post_transformers`
`week11/hyena_lineage.py` (FFTConv). Hand the delta-rule scope to Ch 12; don't absorb it.

**Context when I return:**
- Sequencing is **dependency/readiness-gated, not dated**: 11 → 14 → 16 → 12 → 13 → 15 → 17. The B
  pilot is *blocked on Ch 12/14/16 being authored* (Ch 12 is also B-load-bearing — the delta-rule
  online-learning ODE). C1 is dependency-satisfied by Ch 1–10.
- Ch-11 gate **CLEARED** (2026-06-04): F4/F26/F27 resolved, F6 stale. Record:
  [`audits/2026-06-04_ch11-runway.md`](audits/2026-06-04_ch11-runway.md). Before drafting Ch 11, fill
  `docs/templates/chapter-research-brief.md`. Use `<XRef>` for backward object-refs (STYLE.md §4).
- Control-theory lens for Ch 11–17 = *natural touches only*; a dedicated SSM/control treatment is a
  separate guide (deferred). Run the chapter through `claim-skeptic` before advancing `status:`.
