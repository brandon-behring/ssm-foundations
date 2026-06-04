# Ecosystem checkpoint — 2026-06-04

A lean checkpoint, not a fresh deep audit. It (a) records the current book state, (b) carries forward
the still-open findings as a **Ch-11 authoring gate**, (c) records the design + roadmap decisions taken
this session, and (d) answers the "do we have repo-wide audit tools" question. Status markers are
bracketed text (`[OPEN]` / `[RESOLVED]` / `[DEFERRED]`) per the repo audit convention; carried-forward
findings keep their original `0527-F*` IDs (see `2026-05-29_reconciliation.md`).

Companion to the `/exploring-options` session that produced the ssm-foundations adoptions listed in §5.

---

## 1. Book state (verified 2026-06-04)

- Ch 1–10 = `status: implemented`, shipped + deployed (HEAD `3748738`). Foundations (1–6) + SSM-core
  (7–10: HiPPO → S4/S4D/S5 → Selective → Mamba-3) complete. Ch 11–17 = `planned` stubs.
- **Doc-drift corrected this session:** `README.md` + `CLAUDE.md` had said "Ch 1–6 authored / Ch 7–17
  planned"; `docs/STATUS.md` (verified 2026-05-29) listed ch08/09/10 as 16-line `planned` stubs. All
  three now reflect Ch 1–10 implemented (STATUS.md regenerated → 10 implemented / 7 planned). `[RESOLVED]`

## 2. Open findings carried forward — Ch-11 authoring gate

From the 2026-05-27 deep audit (23 `[OPEN]` at the 2026-05-29 reconciliation). These four are the ones
that should be cleared **before or during** Ch 11, because Ch 11 will exercise the same machinery:

| ID | Finding | Why it gates Ch 11 | Status |
|---|---|---|---|
| 0527-F6 | Mamba-3 absent from `bibliography.bib` despite ~13 uncited mentions | Ch 11 adds Hyena/Performer/RetNet/GLA cites — fix the bib-entry discipline now | `[OPEN]` |
| 0527-F4 | XRef machinery declared but dead (14 IDs, 0 `<XRef>` consumers; linter tolerates) | Decide enforce-or-abandon before adding more chapters' IDs | `[OPEN]` (GH #3) |
| 0527-F27 | torch parity — ch04–06 torch companions absent/`.gitkeep` | Set the torch policy before Ch 11 companions | `[OPEN]` (GH #6) |
| 0527-F26 | no pytest assertion suite for JAX companions | Ch 11 is JAX-first; decide the JAX test bar now | `[OPEN]` (GH #5) |

Other open GH issues: #1 (standards-hardening umbrella), #4 (ch04 Julia in default gate). Full inventory
in `2026-05-29_reconciliation.md`; not re-litigated here.

**New, minor (this session):** `0604-F1` — `README.md:~35` reads "draws on **the user's** research
background" (an authoring artifact; should read "the author's" / "Behring's"). `[OPEN, trivial]` — left
unfixed to honor this session's status-only hygiene scope; flag for the next prose pass.

## 3. Decisions recorded this session

**Control-theory lens (design).** The ecosystem now holds two *verified* bridge dossiers —
`research_bridge_control_rl/synthesis.md` (15 claims) and `research_bridge_ssm_control/synthesis.md`
(27 claims) — plus a Bertsekas/Kalman PDF library (`rl_and_control/`). Decision: **Ch 11–17 take only
natural control-theory touches**; a *dedicated* SSM/control treatment is its **own separate guide**, for
which those dossiers + PDFs are the source material. Not bolted into this book. `[DECIDED]`

**Roadmap reframe (no dated timelines).** Dated targets are too rigid; sequencing is **dependency /
readiness-gated**, not calendar-gated. Authoring order (dependency-driven): **11 → 14 → 16 → 12 → 13 →
15 → 17**. The **B pilot is blocked on Ch 12/14/16 being authored** — a what-blocks-what fact, no date.
Note Ch 12 is *also* B-load-bearing (the delta-rule online-learning ODE underpins B's implicit-vs-explicit
gating analysis), a tension to weigh when sequencing 12 vs 14. **C1 is dependency-satisfied by Ch 1–10.**
`[DECIDED]`

## 4. Repo-wide audit capability (answer to "do we have tools, not just Ch 7–10")

Yes — a true whole-repo audit = two complementary passes over all 10 implemented chapters + infra:
1. **Project consistency** — `/methodology-audit --depth standard|deep` (7 dimensions:
   internal-consistency, claim-grounding, methodology-soundness, decision-record↔impl, doc-code-drift,
   reproducibility, honest-limitations) over CLAUDE.md / README / STYLE.md / every `audits/*` / STATUS.md
   / all frontmatter.
2. **Per-chapter depth** — a multi-agent **Workflow** fanning the now-**five** read-only subagents
   (`chapter-auditor`, `companion-verifier`, `prose-pedagogy-reviewer`, `claim-skeptic`, +
   `citation-link-auditor` once) across Ch 1–10, plus `make lint` + `make validate`.

Documented here as an available follow-on; **not executed this session** (scope was Core + records).

## 5. Adoptions executed this session (ssm-foundations)

- Hygiene: README + CLAUDE.md status corrected; `docs/STATUS.md` regenerated.
- `README.md`: chapter-index table (all 17, with part + status).
- `.claude/agents/claim-skeptic.md`: new adversarial **math-claim** reviewer (the truth-of-claims gap the
  prior four subagents left open); registered in `CLAUDE.md` + `.claude/README.md`.
- Env/context (cheap, non-breaking): `CURRENT_WORK.md`; CLAUDE.md STYLE.md-digest + memory/session
  backlink; `chapter-auditor` reordered to check status-truthfulness first.

## 6. Deferred (decided, not built here)

- **To Ch 11 start:** `docs/DASHBOARD.md` release-trust scoreboard; `docs/templates/chapter-research-brief.md`
  (templates the per-chapter prep doc = lightweight context profile).
- **To a separate `lever_of_archimedes` session:** the guide-making orchestrating pattern + dossier
  template; and a **hub-pattern staleness/currency audit** (are the hub's 13 patterns + 9 guides still
  used / current / good — it is the best-practices reservoir for all spokes, but its advice may be stale).
- **Out of scope:** control-theory wiring; `package.json` bump (v4.8.0 → v4.14.2 available, incl. the
  v4.14.1 `astro dev` fix — `0604-F2 [OPEN, low]`); `post_transformers/` edits; full repo-wide audit run.
