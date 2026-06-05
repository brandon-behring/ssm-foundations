# Current Work — ssm-foundations

One-screen resume context (the `sessions.md` §1 pattern). Update on context switch / session end.
For the durable index, see the Claude Code project-memory `MEMORY.md`.

**Right now:** Ch 11 **authored, merged, and shipped** — PR #18 (`feat/ch11-linear-attention-hyena`)
+ PR #19 (book-wide hygiene) merged to `main` (`23bddd4`). Both upstream toolkit blockers are now
**fixed and bumped**: a branch `chore/toolkit-bump-4.16-theorem-fix` bumps `book-scaffold-astro`
4.8 → **4.16.0**, which (a) fixes the book-wide empty-theorem-label defect (#121, fixed v4.14.3 via
`type→kind` legacy alias) and (b) unblocks `<XRef>` (#120, fixed v4.9.0). On that branch the 6
kind-less theorems (ch10×3, ch11×3) get `type="theorem"` and ch11's 10 cross-refs are migrated to
`<XRef>`. **Bump PR is CI-green and awaiting your merge** — that merge is the deploy that fixes every
theorem label in production. Ch 1–11 `implemented`.

**Why:** Foundations + the SSM spine (HiPPO → S4/S4D/S5 → Selective → Mamba-3) + the first beyond-SSM
chapter (11, linear attention + Hyena) are done. Part IV (beyond-ssm) continues.

**Next step:** after the bump PR merges, **Ch 14 — *Hybrid architectures and gating mechanisms*** (per
the 11 → 14 → 16 order). Start with `/exploring-options` (chapter-authoring playbook), then
companions-first. Ch 14 + Ch 16 are authored against the B pilot's two-timescale-benchmark needs.

**Context when I return:**
- Sequencing is **dependency/readiness-gated, not dated**: 11 → 14 → 16 → 12 → 13 → 15 → 17. The B
  pilot is *blocked on Ch 12/14/16 being authored* (Ch 12 is also B-load-bearing — the delta-rule
  online-learning ODE). C1 is dependency-satisfied by Ch 1–10.
- **Toolkit bugs resolved (2026-06-05):** theorem labels render again (bump to ≥4.14.3) and `<XRef>`
  compiles (≥4.9.0). STYLE §4's XRef-forward policy is now live and exercised in ch11 — use `<XRef>`
  for backward object-refs in Ch 12+. The book is uniform on `type=` for theorem kinds.
- Control-theory lens for Ch 11–17 = *natural touches only*; a dedicated SSM/control treatment is a
  separate guide (deferred). Run each chapter through `claim-skeptic` before advancing `status:`.
