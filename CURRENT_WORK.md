# Current Work — ssm-foundations

One-screen resume context (the `sessions.md` §1 pattern). Update on context switch / session end.
For the durable index, see the Claude Code project-memory `MEMORY.md`.

**Right now:** No chapter in flight. Ch 1–10 are `implemented`, shipped + deployed — the foundations
(1–6) and SSM-core (7–10) lines are complete (HEAD `3748738`).

**Why:** Foundations + the SSM spine (HiPPO → S4/S4D/S5 → Selective → Mamba-3) are done. Part IV
(beyond-ssm) is next, opening with Ch 11.

**Next step:** Ch 11 — *Linear attention and Hyena*. Start with `/exploring-options` (per the
chapter-authoring playbook), then companions-first. High predecessor reuse: `post_transformers`
`week11/hyena_lineage.py` (FFTConv). Hand the delta-rule scope to Ch 12; don't absorb it.

**Context when I return:**
- Sequencing is **dependency/readiness-gated, not dated**: 11 → 14 → 16 → 12 → 13 → 15 → 17. The B
  pilot is *blocked on Ch 12/14/16 being authored* (Ch 12 is also B-load-bearing — the delta-rule
  online-learning ODE). C1 is dependency-satisfied by Ch 1–10.
- Open Ch-11-gate findings to clear: **F6** (Mamba-3 missing from `bibliography.bib` + 13 uncited),
  **F4** (XRef machinery declared but unused), **F27** (torch parity ch04–06), **F26** (no JAX pytest).
  See [`audits/2026-06-04_ecosystem_checkpoint.md`](audits/2026-06-04_ecosystem_checkpoint.md).
- Control-theory lens for Ch 11–17 = *natural touches only*; a dedicated SSM/control treatment is a
  separate guide (deferred). Run the chapter through `claim-skeptic` before advancing `status:`.
