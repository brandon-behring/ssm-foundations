# ssm-foundations — Standards & Infrastructure Audit vs. post_transformers

**Audit date:** 2026-05-25
**Audience:** repo owner (single-maintainer book repo)
**Scope:** full repo — tracked surface plus sibling-repo comparison against
[`brandon-behring/post_transformers`](https://github.com/brandon-behring/post_transformers)
(the 21-week research curriculum predecessor) — as of 2026-05-25
**Supersedes:** none (this is the inaugural canonical audit)

---

## 0. Executive Summary

ssm-foundations is **1 day old as a public artifact** but already contains six
substantially-authored chapters (Ch 1–6, ~1900 lines of MDX, 18 exercises, 16
bibliography entries) with strong content rigor inherited from the sibling
research repo. The standards/infrastructure layer has not kept pace: there is
no `LICENSE`, no `CONTRIBUTING.md`, no `audits/` (until now), no `scripts/`,
no pre-commit, no Makefile, no CI/CD, no consolidated authoring style guide,
and no companion-code linting parity across Python / Julia / torch.

This audit identifies **three compounding debts** (§2), enumerates **18
findings** (§3) ranked across Track A (quick wins, ≤1 session), Track B
(substantive remediation, GH-issue tracked), and Track C (structural debt,
deferred until repo maturity warrants). It adopts the F-numbered priority-table
format from
[`post_transformers/audits/archive/2026-04-11_repo_maintainability_audit.md`](https://github.com/brandon-behring/post_transformers/blob/main/audits/archive/2026-04-11_repo_maintainability_audit.md),
with a deliberate deviation: status markers use bracketed text labels
(`[open]` / `[tracked]` / `[fixed]` / `[pilot-blocked]`) instead of emoji, for
grep / `git log -S` / text-only-render durability.

**Top three Track-A quick wins to execute next session:**

1. **F1** — fix CLAUDE.md status drift (says Ch 1–3 `scaffolded`, Ch 4–17
   `planned`; reality is Ch 1–6 all `implemented`).
2. **F8** — close Julia + torch companion-rigor gap before the C1 pilot
   lands on 2026-06-01 (`JuliaFormatter.toml`, per-chapter `runtests.jl`,
   torch optional extras in `companions/_shared/pyproject.toml`).
3. **F2** — add `lever_of_archimedes/patterns/` hub-reference block to
   CLAUDE.md, modeled on
   [`post_transformers/CLAUDE.md:114–121`](https://github.com/brandon-behring/post_transformers/blob/main/CLAUDE.md#L114-L121).

**Track A follow-on** (added post-F8 execution): **F19** — correct
Ch 6 RK4-vs-symplectic energy-drift magnitude claims (six contradictions
between prose, exercise solution, demo, and the actual figure parameters).

---

## 1. Audit Context & Scope

The repo crossed a threshold on the day it shipped: from "scaffolded book
template" (2026-05-23) to "6 substantively authored chapters + Plan 3 launched
ahead of trigger" (per
[`post_transformers/notes/niche_decision_2026_05_24.md`](https://github.com/brandon-behring/post_transformers/blob/main/notes/niche_decision_2026_05_24.md)).
Chapter prose is rigorous and consistent across Ch 1–6; companion code in
JAX is professionally tooled. But onboarding documentation, build automation,
companion-rigor parity (Julia + torch), and content-quality enforcement have
not kept pace with the authoring velocity.

**The maintenance question this audit answers:**

| Era | Question | Answer |
|-----|----------|--------|
| 2026-05-23 (scaffold landed) | "Will the build pipeline work?" | Yes — book-scaffold-astro v4.2.0 inherits validation, bib/labels build, Cloudflare deploy |
| 2026-05-24 (Plan 3 launch) | "Are Ch 1–6 worth shipping?" | Yes — they're authored to research-monograph standards |
| 2026-05-25 (this audit) | **"Is this sustainable as Ch 7–17 author over 2026-Q3/Q4 with the C1 + B pilots producing empirical claims?"** | **Not yet — three debts are compounding faster than authoring is filling them** |

This audit covers **10 dimensions**: repo structure & governance, build &
deploy automation, content quality enforcement, linting & formatting, companion
code standards, testing infrastructure, documentation, issue/PR hygiene,
reproducibility, cross-repo references & hub integration, plus a derived
**11th dimension** — *content authoring standards* (callout taxonomy, bibkey
discipline, theorem-label conventions, port-credit headers) — that surfaced
when comparing actual chapter MDX against
[`post_transformers/guides/STANDARDS.md`](https://github.com/brandon-behring/post_transformers/blob/main/guides/STANDARDS.md).

**Method notes:**

- Evidence is from direct file reads of the repo as of 2026-05-25 plus
  read-only inspection of post_transformers at the same date.
- All post_transformers URLs are absolute GitHub URLs pinned to `main`, per
  the documented cross-repo link convention (CLAUDE.md §"Cross-repo link
  convention").
- Recommendations are designed to be transferable from post_transformers; §5
  explicitly enumerates patterns that should **not** transfer.

**Format deviation from the April 11 template:** Status markers in §3 use
bracketed text labels (`[open]`, `[tracked]`, `[fixed]`, `[pilot-blocked]`)
instead of emoji. Rationale: emoji status markers don't survive `grep`,
`git log -S`, or text-only renderings reliably. Bracketed text labels are
durable artifact-tracking tokens. Future audits in this repo should follow
the same convention.

---

## 2. Core Thesis: Three Compounding Debts (by compounding rate)

1. **Authoring-convention debt** (compounds fastest, with each chapter
   authored) — Ch 1–6 have lived conventions (theorem/definition/proposition
   component usage, exercise structure with §X.8 Exercises + §X.9 Full
   solutions, bibkey format `<firstauthor><year><firstword>`, port-credit
   headers, KaTeX macros) that are not codified anywhere. Ch 5 already
   shows drift: its exercise section is `§5.7` while Ch 1–4 + Ch 6 use `§X.8`.
   Each new chapter authored without these documented increases drift cost.
   Findings: **F5** (no STYLE.md), F6 (bibkey lint), F7 (theorem-label lint),
   F4 (port-credit convention).
2. **Companion-rigor asymmetry** (compounds with each Julia/torch landing) —
   Python companions are strictly tooled
   (`companions/_shared/pyproject.toml`: black 100-char, ruff with E/F/W/I/N/UP/B/SIM,
   mypy strict). Julia has 3 `Project.toml` files (ch04, ch05, ch06) but no
   `JuliaFormatter.toml`, no `runtests.jl`, no test discipline. Torch
   directories are entirely missing across all chapters. The C1 pilot is
   anchored on Julia symplectic integrators and starts on 2026-06-01 (6
   days from audit date); pilot-relevant code will land in
   `companions/ch06/julia/symplectic_methods.jl` without lint/test
   infrastructure. Findings: **F8** (Julia + torch rigor parity).
3. **Truthfulness debt** (compounds with memory decay) —
   [`CLAUDE.md:5`](../CLAUDE.md) says "Chapters 1–3 are next-up for authoring;
   Ch 4–17 are stubbed `planned`"; [`CLAUDE.md:42`](../CLAUDE.md) says
   "Current state (2026-05-24): Ch 1–3 = `scaffolded` (authoring imminent);
   Ch 4–17 = `planned`";
   [`companions/README.md:15`](../companions/README.md) says "empty
   per-language subdirs for Ch 1–3 (Phase 3 will populate). Ch 4–17
   directories will be created on-demand". **Reality (2026-05-25):** all
   Ch 1–6 frontmatter is `status: implemented`; Ch 1–3 jax subdirs each have
   2–3 files; Ch 4–6 have both jax and Julia companions. The repo's
   onboarding docs lie to fresh readers. Low cost to fix; high cost if not
   fixed — every external collaborator or future-self reading these claims
   will form wrong expectations. Findings: **F1**, **F19** (added 2026-05-25 after F8 surfaced a Ch 6 content-correctness contradiction).

Everything else (build automation, governance docs, CI) is secondary to
these three for the *current* state of the repo. Several of them are
nonetheless quick wins worth doing soon, batched in Track A.

---

## 3. All-Findings Priority Table

**Status vocabulary** (text labels, deliberate deviation from
post_transformers' April 11 audit which uses emojis):

- `[open]` — newly found, no remediation in progress
- `[tracked]` — GH issue created, queued for execution
- `[fixed]` — remediated in this session or prior
- `[pilot-blocked]` — gated on C1 or B pilot outcome; unblock condition stated in §4

**Track definitions:**

- **Track A**: quick wins, low cost, can be fixed in the audit session or
  next. Marked `[fixed]` inline once executed. *In this inaugural audit, all
  Track A remediation is deferred to a follow-up session per the scope
  agreed at plan time — the Track A designation signals urgency, not
  in-session execution.*
- **Track B**: substantive remediation, multi-session work, tracked via GH
  issue.
- **Track C**: structural debt, long-term scaffolding (LICENSE, CI, ADRs,
  governance), deferred until repo maturity warrants.

**Pilot-imminent gate** (referenced from §1): any finding whose unblock event
— pilot start, chapter authoring landing, deployment milestone — falls within
≤14 days of audit date is promoted to Track A regardless of remediation
effort. As of 2026-05-25, the C1 pilot starts ~2026-06-01 (6 days), so F4
and F8 are promoted under this rule. The gate is documented so future audits
can apply the same logic consistently.

| #   | Severity     | Area                  | Finding                                                                                                          | Track | Lens     | GH  | Status |
|-----|--------------|-----------------------|------------------------------------------------------------------------------------------------------------------|-------|----------|-----|--------|
| F1  | **CRITICAL** | Truthfulness          | `CLAUDE.md:5,42` + `companions/README.md:15` claim Ch 1–3 scaffolded / Ch 4–17 planned; reality is Ch 1–6 all `implemented` | A     | Maint    | #1  | `[fixed]` |
| F2  | IMPORTANT    | Hub integration       | `CLAUDE.md` has no `lever_of_archimedes/patterns/` reference block (post_transformers/CLAUDE.md:114–121 does)     | A     | Maint    | #1  | `[fixed]` |
| F3  | IMPORTANT    | Audit cadence         | No `audits/` directory exists                                                                                    | A     | Maint    | —   | `[fixed]` |
| F4  | MINOR*       | Content authoring     | Port-credit convention undocumented; only `companions/_shared/plot_utils.py:1–16` has a credit header            | A*    | Authoring | #1  | `[fixed]` |
| F5  | **CRITICAL** | Content authoring     | No consolidated authoring style guide (Ch 1–6 conventions un-codified; Ch 5 already shows drift `§5.7` vs `§X.8`) | B     | Authoring | #1  | `[fixed]` |
| F6  | IMPORTANT    | Content authoring     | No bibkey lint (16-entry `bibliography.bib`, bibkey format `<firstauthor><year><firstword>` unenforced)         | B     | Authoring | TBD | `[open]` |
| F7  | IMPORTANT    | Content authoring     | No theorem cross-ref label lint (`id="thm:chXX:slug"` convention implicit)                                       | B     | Authoring | TBD | `[open]` |
| F8  | **CRITICAL** | Companion rigor       | Julia + torch companions lack rigor Python has; C1 pilot starts 2026-06-01 → promoted to Track A                 | A*    | Pilot    | #1  | `[fixed]` |
| F9  | IMPORTANT    | Quality enforcement   | No `.pre-commit-config.yaml`                                                                                     | B     | Authoring | TBD | `[open]` |
| F10 | IMPORTANT    | Build automation      | No `Makefile` (npm scripts only; no aliasing for status snapshot, bibkey check, xref check)                      | B     | Maint    | TBD | `[open]` |
| F11 | IMPORTANT    | Status reporting      | No `docs/STATUS.md` snapshot (per-chapter status, word count, citation count, exercise/companion coverage)        | B     | Maint    | #1  | `[fixed]` |
| F12 | IMPORTANT    | Governance            | `LICENSE` + `LICENSE-CONTENT` split absent (post_transformers waited 8 weeks before adding these)                 | C     | Prod     | TBD | `[open]` |
| F13 | IMPORTANT    | Governance            | `CONTRIBUTING.md` absent                                                                                         | C     | Prod     | TBD | `[open]` |
| F14 | IMPORTANT    | CI/CD                 | No `.github/workflows/validate.yml`                                                                              | C     | Prod     | TBD | `[open]` |
| F15 | IMPORTANT    | Issue/PR hygiene      | No `.github/ISSUE_TEMPLATE/` or `pull_request_template.md`                                                       | C     | Prod     | TBD | `[open]` |
| F16 | MINOR        | Editor consistency    | No `.editorconfig`                                                                                               | C     | Authoring | TBD | `[open]` |
| F17 | IMPORTANT    | Content authoring     | Practice tags (`\tagofficial`/`\tagpractitioner`/`\tagconv` in post_transformers/guides/STANDARDS.md) have no MDX equivalent | C     | Pilot    | TBD | `[pilot-blocked]` |
| F18 | IMPORTANT    | Hub integration       | `precision.md` pattern from `lever_of_archimedes/patterns/` not adopted                                          | C     | Pilot    | TBD | `[pilot-blocked]` |
| F19 | IMPORTANT    | Content correctness   | Ch 6 prose + companion overstate RK4 vs symplectic energy drift magnitudes by 3–4 orders at cited parameters; figures' actual parameters mismatch the caption text | A     | Authoring | #1  | `[fixed]` |

\* F4 (MINOR) and F8 (CRITICAL) were promoted to Track A via the
pilot-imminent gate even though Track A normally implies low effort. F8's
effort is moderate (≈50 lines across `JuliaFormatter.toml`, `runtests.jl`
template, pyproject extras) but its urgency forces the promotion.

### Coverage check (one bullet per lens)

- **Production readiness**: F12, F13, F14, F15 — all Track C (deferred).
  Coverage adequate; no Track A items because production hardening is
  appropriate later.
- **Authoring velocity**: F4, F5, F6, F7, F9, F16 — F4 in Track A, F5–F7 in
  Track B, F9 in Track B, F16 in Track C. Strong coverage matching the
  "authoring-convention debt" thesis.
- **Pilot integration**: F8 (Track A, promoted), F17, F18 (Track C,
  pilot-blocked). F8 covers immediate companion-rigor need; F17–F18 wait for
  empirical claims from pilots.
- **Long-term maintenance**: F1, F2, F3, F10, F11 — 3 Track A (F1, F2, F3)
  plus 2 Track B (F10, F11). Strong coverage.

Each thesis debt in §2 is supported by ≥2 findings:
- *Authoring-convention debt* → F4, F5, F6, F7
- *Companion-rigor asymmetry* → F8 (with F17 deferred)
- *Truthfulness debt* → F1, F19

---

## 4. Per-Finding Detail

### F1 — CLAUDE.md / companions/README.md status drift

**Severity:** CRITICAL · **Track:** A · **Lens:** Long-term maintenance ·
**Status:** `[fixed]` (umbrella issue #1) · **Contributes to thesis debt:** Truthfulness debt (§2.3)

**Evidence:**

- [`CLAUDE.md:5`](../CLAUDE.md): "Chapters 1–3 are next-up for authoring;
  Ch 4–17 are stubbed `planned`."
- [`CLAUDE.md:42`](../CLAUDE.md): "Current state (2026-05-24): Ch 1–3 =
  `scaffolded` (authoring imminent); Ch 4–17 = `planned`."
- [`companions/README.md:15`](../companions/README.md): "Current state
  (2026-05-24): empty per-language subdirs for Ch 1–3 (Phase 3 will
  populate). Ch 4–17 directories will be created on-demand as chapters are
  authored."
- Reality (verified via `grep status: src/content/chapters/*.mdx`): all of
  `ch01-linear-odes.mdx`, `ch02-stability-theory.mdx`, `ch03-linear-algebra.mdx`,
  `ch04-discretization.mdx`, `ch05-stability-regions.mdx`,
  `ch06-implicit-and-symplectic.mdx` are `status: implemented`.
- Reality (verified via `ls companions/chXX/jax`): Ch 1 jax has 3 files; Ch
  2 jax has 2; Ch 3 jax has 2; Ch 4 jax has 2 + julia 1; Ch 5 jax 2 +
  julia 1; Ch 6 jax 2 + julia 2.

**Recommendation:**

- Update `CLAUDE.md:5` "Status" line and `:42` "Current state" line to:
  "Current state (2026-05-25): Ch 1–6 = `implemented`; Ch 7–17 = `planned`."
- Update `companions/README.md:15` to: "Current state (2026-05-25): Ch 1–6
  populated (JAX across all six, Julia in Ch 4–6); Ch 7–17 directories
  created on-demand as chapters author."
- Add a follow-up reminder to keep these in sync as Ch 7+ author — see F11
  (auto-generated `docs/STATUS.md` snapshot) for the durable fix.

**Why Track A:** trivial edit (3 lines across 2 files) but high consequence
(the repo lies to readers at the top of CLAUDE.md right now).

---

### F2 — CLAUDE.md missing lever_of_archimedes hub-reference block

**Severity:** IMPORTANT · **Track:** A · **Lens:** Long-term maintenance ·
**Status:** `[fixed]` (umbrella issue #1) · **Contributes to thesis debt:** none directly; hub
integration is orthogonal to the three debts

**Evidence:**

- [`post_transformers/CLAUDE.md:114–121`](https://github.com/brandon-behring/post_transformers/blob/main/CLAUDE.md#L114-L121)
  has an explicit "Platform Integration" section pointing to
  `~/Claude/lever_of_archimedes/` with named references to `git.md`,
  `testing.md`, `sessions.md`.
- ssm-foundations/CLAUDE.md: no equivalent block. Conventions are followed
  in practice (e.g., commit messages use conventional-commits format from
  `git.md`) but are not documented in the project's onboarding doc.

**Recommendation:**

- Add a "Platform Integration" or "Hub Pattern References" section near the
  end of `CLAUDE.md`, modeled on
  [`post_transformers/CLAUDE.md:114–121`](https://github.com/brandon-behring/post_transformers/blob/main/CLAUDE.md#L114-L121).
- Include explicit links to:
  - `~/Claude/lever_of_archimedes/patterns/git.md` (commit format)
  - `~/Claude/lever_of_archimedes/patterns/sessions.md` (session workflow)
  - `~/Claude/lever_of_archimedes/patterns/deploy_subdomain_brandon_behring_dev.md`
    (deployment convention — directly applies; book is deployed at
    `ssm-foundations.brandon-behring.dev` per CLAUDE.md §"Build + deploy")
- Cross-ref: substantive adoption of `testing.md` is covered by F8 (Julia +
  torch companion rigor); `precision.md` adoption is deferred via F18.

**Why Track A:** small CLAUDE.md edit (~10 lines).

---

### F3 — No `audits/` directory existed

**Severity:** IMPORTANT · **Track:** A · **Lens:** Long-term maintenance ·
**Status:** `[fixed]` · **Contributes to thesis debt:** Truthfulness debt
(§2.3, indirectly — audits are how truthfulness gets re-checked)

**Evidence:**

- `ls audits/` returned "no such directory" at audit start.
- post_transformers established its audit cadence early (April 11, 2026 —
  approximately 2 weeks after repo creation) and has used the dated
  `audits/YYYY-MM-DD_*.md` pattern with archive policy ever since (see
  [`post_transformers/audits/README.md`](https://github.com/brandon-behring/post_transformers/blob/main/audits/README.md)).

**Recommendation:**

- Create `audits/README.md` (modeled on
  [`post_transformers/audits/README.md`](https://github.com/brandon-behring/post_transformers/blob/main/audits/README.md))
  documenting the archive policy: one canonical at top level, all prior
  audits in `archive/`, dated filename convention `YYYY-MM-DD_short_scope.md`.
- This audit (`2026-05-25_standards_vs_post_transformers.md`) is the
  inaugural canonical.

**Why `[fixed]`:** the act of writing this audit + `audits/README.md` IS
the remediation. No follow-up work needed for this finding.

---

### F4 — Port-credit convention undocumented (pilot-imminent promotion)

**Severity:** MINOR (promoted to Track A via pilot-imminent gate) ·
**Track:** A · **Lens:** Authoring velocity · **Status:** `[fixed]` (umbrella issue #1) ·
**Contributes to thesis debt:** Authoring-convention debt (§2.1)

**Evidence:**

- [`companions/_shared/plot_utils.py:1–16`](../companions/_shared/plot_utils.py)
  is explicitly credited as "Minimal port of
  ``post_transformers/guides/shared/plot_utils.py``" in its docstring.
- This is the only ported file in the repo with a credit header. The
  convention is implicit (one example) rather than documented (a policy).
- C1 pilot lands on Julia symplectic code starting 2026-06-01 (6 days from
  audit). Likely it will draw from
  `post_transformers/experiments/julia/discretization_atlas/` and similar.
  Without a documented convention, derived code may land without credits,
  losing provenance.

**Recommendation:**

- Add a 1-paragraph section to `companions/README.md` documenting the
  port-credit header convention:
  ```
  ## Port-credit convention
  When a companion file is derived (in whole or substantively) from code in
  post_transformers/, the first docstring/comment block must cite the source
  with an absolute GitHub URL pinned to `main`. Example:
  `companions/_shared/plot_utils.py:1–16`.
  ```
- Optional small extension: a one-line `grep -L "post_transformers" companions/**/*.py`
  check could verify ported files declare their lineage; defer this to F6/F7
  lint-script scope.

**Why Track A (via pilot-imminent gate):** convention must exist before C1
pilot code is ported in.

---

### F5 — No consolidated authoring style guide

**Severity:** CRITICAL · **Track:** B · **Lens:** Authoring velocity ·
**Status:** `[fixed]` (umbrella issue #1) · **Contributes to thesis debt:** Authoring-convention
debt (§2.1) — primary driver

**Evidence:**

- Ch 1–6 share consistent component usage (`<Theorem type="definition">`,
  `<Theorem type="proposition">`, `<Cite key="">`, `<MarginNote>`,
  `<Figure>` with `src=/figures/chXX/...png`), but the conventions are
  emergent, not documented.
- Drift is already starting: Ch 1–4 + Ch 6 use `## X.8 Exercises` and
  `## X.9 Full solutions`; Ch 5 uses `## 5.7 Exercises` and `## 5.8 Full
  solutions`. Without a style guide, Ch 7–17 will diverge further.
- post_transformers has a 23KB consolidated style guide at
  [`guides/STANDARDS.md`](https://github.com/brandon-behring/post_transformers/blob/main/guides/STANDARDS.md)
  (commit `d46ddcb` on 2026-04-18: "consolidate scattered authoring standards"
  — note that it was an emergent consolidation, not a designed-up-front
  artifact).

**Recommendation:**

- Create `STYLE.md` (or `AUTHORING.md`) at repo root consolidating Ch 1–6
  conventions. Sections, adapted for MDX:
  - **Chapter skeleton**: §X.1 Motivation → §X.2…X.7 content → §X.8
    Exercises → §X.9 Full solutions → §X.10 Companions and next chapter.
    Standardize §X.8/§X.9 numbering (renumber Ch 5 in a follow-up).
  - **Component vocabulary**: `<Theorem type="definition|proposition|theorem">`
    with `id="thm:chXX:slug"` (see F7), `<Figure>`, `<Cite>`, `<MarginNote>`,
    `<BlockedByCallout>`. Document expected props per component.
  - **Math notation**: KaTeX macros (currently in chapter frontmatter
    comments). Decide: keep per-chapter, or extract to a shared
    `src/styles/ssm-macros.ts` consumed by all chapters.
  - **Bibkey convention**: `<firstauthor><year><firstword>` (matches
    post_transformers); see F6 for lint.
  - **Citation style**: `<Cite key="...">` inline; full bibliography
    auto-generated via `npm run build:bib`.
  - **Figure paths**: `/figures/chXX/<slug>.png` produced by
    `companions/chXX/jax/<script>.py` (or julia variant). Document the
    `PYTHONPATH=. python companions/...` pattern.
  - **Exercise pattern**: 6–7 problems per chapter, mixing computation and
    theory; short ones get inline `<details>` solutions; long ones get
    `## X.9 Full solutions to theory exercises`.
  - **Port-credit headers** (see F4): every derived file cites source with
    absolute GitHub URL pinned to `main`.

**Why Track B:** non-trivial (~2 hours to extract, organize, validate from
Ch 1–6 prose) but no urgent deadline. Filed as GH issue under umbrella.

---

### F6 — No bibkey lint

**Severity:** IMPORTANT · **Track:** B · **Lens:** Authoring velocity ·
**Status:** `[open]` · **Contributes to thesis debt:** Authoring-convention
debt (§2.1)

**Evidence:**

- `bibliography.bib` has 16 entries. Bibkeys follow `<firstauthor><year><firstword>`
  format (e.g., `gu2020hippo`, `gu2022s4`, `dao2024mamba2`, `hairer1993ordinary`).
- This format is byte-identical with post_transformers'
  `guides/shared/references.bib` — ssm-foundations entries are a strict
  subset of post_transformers'.
- No script enforces this format. A typo (`Gu2022_S4` or `s4-2022`) on the
  next added entry would silently introduce drift.
- No script enforces that every `<Cite key="...">` in MDX resolves to an
  existing entry in `bibliography.bib`.

**Recommendation:**

- Add `scripts/check-bibkeys.mjs` validating two things:
  1. Every bibkey in `bibliography.bib` matches regex
     `^[a-z]+\d{4}[a-z]+$` (firstauthor lowercase + 4-digit year +
     firstword lowercase).
  2. Every `<Cite key="...">` in `src/content/chapters/*.mdx` resolves to
     an existing bibkey in `bibliography.bib`.
- Wire into `npm run validate` (currently calls `book-scaffold validate`;
  this script runs alongside). Cross-ref F9 for pre-commit gating.

**Why Track B:** ~30 minutes to write, but worth GH-issue tracking so the
implementation lands with explicit acceptance criteria.

---

### F7 — No theorem cross-ref label lint

**Severity:** IMPORTANT · **Track:** B · **Lens:** Authoring velocity ·
**Status:** `[open]` · **Contributes to thesis debt:** Authoring-convention
debt (§2.1)

**Evidence:**

- Chapters use `id="..."` on `<Theorem>` and `<Figure>` to anchor
  cross-references (per CLAUDE.md §"Where things live": `id="..."` →
  `src/data/labels.json` via `npm run build:labels`).
- Convention is implicitly `id="thm:chXX:<slug>"` or `id="fig:chXX:<slug>"`,
  but not documented or enforced.
- A typo (`thm:ch01-energy` vs `thm:ch01:energy`) would produce a broken
  cross-reference at build time but no descriptive error.

**Recommendation:**

- Add `scripts/check-xref-labels.mjs` validating:
  1. Every `id="..."` on `<Theorem>` matches `^thm:ch\d{2}:[a-z0-9-]+$`.
  2. Every `id="..."` on `<Figure>` matches `^fig:ch\d{2}:[a-z0-9-]+$`.
  3. Every `<XRef target="...">` resolves to a known label.
- Wire into `npm run validate` alongside F6.

**Why Track B:** companion of F6; share GH issue if tracked.

---

### F8 — Julia + torch companion rigor gap (pilot-imminent promotion)

**Severity:** CRITICAL (promoted to Track A via pilot-imminent gate) ·
**Track:** A · **Lens:** Pilot integration · **Status:** `[fixed]` (umbrella issue #1) ·
**Contributes to thesis debt:** Companion-rigor asymmetry (§2.2) — primary
driver

**Evidence:**

- Python companions:
  [`companions/_shared/pyproject.toml`](../companions/_shared/pyproject.toml)
  declares black 100-char (line 26), ruff with rules E/F/W/I/N/UP/B/SIM
  (line 34), mypy strict (line 39).
- Julia companions: 3 `Project.toml` files exist
  (`companions/ch04/julia/Project.toml`, `companions/ch05/julia/Project.toml`,
  `companions/ch06/julia/Project.toml`). No `JuliaFormatter.toml` anywhere.
  No `runtests.jl` anywhere. No `Manifest.toml` (probably correctly
  gitignored, but no documented policy).
- Torch companions: no `companions/chXX/torch/` directories exist for any
  chapter.
- C1 pilot timeline (per
  [`post_transformers/notes/niche_decision_2026_05_24.md`](https://github.com/brandon-behring/post_transformers/blob/main/notes/niche_decision_2026_05_24.md)):
  empirical work starts ~2026-06-01 (6 days from audit date) on Julia
  symplectic integrators. Code will land in
  `companions/ch06/julia/symplectic_methods.jl`.
- `companions/ch06/julia/symplectic_methods.jl` already exists as a stub.

**Recommendation:**

- **Track A (pilot-imminent):** before 2026-06-01, add:
  - `companions/_shared/JuliaFormatter.toml` — minimal config matching
    SciML style (92-char lines per post_transformers convention), enforcing
    consistent indentation, import order, function spacing. ~20 lines.
  - `companions/ch04/julia/runtests.jl`, `companions/ch05/julia/runtests.jl`,
    `companions/ch06/julia/runtests.jl` — minimal `@testset` template, ~30
    lines each. Initial assertions can be smoke tests (function exists,
    returns expected shape).
  - `companions/_shared/pyproject.toml` extension: add `torch` optional
    extras dependency block, ~5 lines, so future torch companions inherit
    Python tooling.
- Defer to Track B (post-pilot): adding `[tool.juliaformatter]` config in
  `Project.toml` for per-chapter overrides; integrating `julia runtests.jl`
  into pre-commit via F9.

**Why Track A (via pilot-imminent gate):** rigor must arrive before code does.

---

### F9 — No `.pre-commit-config.yaml`

**Severity:** IMPORTANT · **Track:** B · **Lens:** Authoring velocity ·
**Status:** `[open]` · **Contributes to thesis debt:** Authoring-convention
debt (§2.1)

**Evidence:**

- No `.pre-commit-config.yaml` exists at repo root.
- post_transformers'
  [`.pre-commit-config.yaml`](https://github.com/brandon-behring/post_transformers/blob/main/.pre-commit-config.yaml)
  is Makefile-routed (philosophy: gate, `make check`, and any future CI
  invocation use byte-identical commands). It runs `make lint` (ruff + black
  on staged Python) and `make nbtest-ci` (notebook regression gate).

**Recommendation:**

- After F6, F7, F10 land, add `.pre-commit-config.yaml` with hooks:
  - `validate` — runs `npm run validate` (book-scaffold's validation) on
    staged `.mdx` files.
  - `check-bibkeys` — runs `node scripts/check-bibkeys.mjs` on staged
    `bibliography.bib` or `.mdx`.
  - `check-xref-labels` — runs `node scripts/check-xref-labels.mjs` on
    staged `.mdx`.
  - `format-python` — black + ruff on staged `companions/**/*.py`.
  - `format-julia` — JuliaFormatter check on staged `companions/**/*.jl`
    (once F8 lands).
- Route through Makefile (F10) so commands are byte-identical with CI.

**Why Track B:** depends on F6/F7/F8/F10 landing first.

---

### F10 — No Makefile

**Severity:** IMPORTANT · **Track:** B · **Lens:** Long-term maintenance +
Authoring velocity · **Status:** `[open]` · **Contributes to thesis debt:**
none directly; supports F9, F11

**Evidence:**

- No `Makefile` at repo root.
- `package.json` has `scripts`: `predev`, `prebuild`, `build:bib`,
  `build:labels`, `build:figures`, `build:notebooks`, `validate`, `dev`,
  `build`, `preview` — but no aliasing for status snapshots, bibkey checks,
  or xref checks.
- post_transformers'
  [`Makefile`](https://github.com/brandon-behring/post_transformers/blob/main/Makefile)
  is 215 lines with 20+ targets; pre-commit + `make check` use byte-identical
  commands.

**Recommendation:**

- Add a minimal `Makefile` at repo root with targets:
  - `make validate` — alias for `npm run validate`.
  - `make check-bibkeys` — `node scripts/check-bibkeys.mjs` (once F6 lands).
  - `make check-xrefs` — `node scripts/check-xref-labels.mjs` (once F7 lands).
  - `make status-snapshot` — runs `scripts/generate-status.mjs` to
    regenerate `docs/STATUS.md` (once F11 lands).
  - `make check` — composes `validate` + `check-bibkeys` + `check-xrefs` +
    `status-snapshot`. Fast-fail.
- Document in `CLAUDE.md` §"Build + deploy".

**Why Track B:** depends on scripts/ landing first.

---

### F11 — No `docs/STATUS.md` snapshot

**Severity:** IMPORTANT · **Track:** B · **Lens:** Long-term maintenance ·
**Status:** `[fixed]` (umbrella issue #1) · **Contributes to thesis debt:** Truthfulness debt
(§2.3) — durable fix

**Evidence:**

- No `docs/` directory. No `STATUS.md`.
- F1 (CLAUDE.md status drift) is symptomatic of having no auto-generated
  truth source. post_transformers solved this with
  [`docs/STATUS.md`](https://github.com/brandon-behring/post_transformers/blob/main/docs/STATUS.md)
  auto-regenerated via `make status-snapshot`, with a Makefile staleness
  gate (≤14 days; per
  [`post_transformers/Makefile`](https://github.com/brandon-behring/post_transformers/blob/main/Makefile)
  `status-check` target).

**Recommendation:**

- Add `scripts/generate-status.mjs` that walks `src/content/chapters/*.mdx`
  and emits a Markdown table with columns per chapter:
  - status (from frontmatter)
  - lines (file length)
  - exercises (count of `### Solution to Exercise` headings)
  - citations (count of distinct `<Cite key="">` keys)
  - figures (count of `<Figure>` components)
  - companion completeness (count of files in `companions/chXX/*/`)
- Output goes to `docs/STATUS.md` with a `verified 2026-MM-DD` header.
- `make status-snapshot` regenerates it; `make status-check` fails if
  `verified` date is >14 days old.

**Why Track B:** durable fix for the truthfulness debt (F1). Implementation
is non-trivial (~1 hour) but contained.

---

### F12 — LICENSE + LICENSE-CONTENT split absent

**Severity:** IMPORTANT · **Track:** C · **Lens:** Production readiness ·
**Status:** `[open]` · **Contributes to thesis debt:** none

**Evidence:**

- No `LICENSE` or `LICENSE-CONTENT` files at repo root.
- post_transformers waited 8 weeks after repo creation (commit `ebdf96b` on
  2026-05-18: "Phase 6 — public-repo hardening (license, gitleaks,
  CONTRIBUTING)") — 2 months from repo start to LICENSE landing.

**Recommendation (deferred to Track C):**

- When the repo approaches a pre-1.0 milestone or has its first external
  reader / contributor, add:
  - `LICENSE` — MIT (covers code: companion code, scripts, config).
  - `LICENSE-CONTENT` — CC BY 4.0 (covers prose: chapter MDX, exercises,
    notes).
- Use post_transformers'
  [`CONTRIBUTING.md:17–21`](https://github.com/brandon-behring/post_transformers/blob/main/CONTRIBUTING.md#L17-L21)
  as a template for explaining the split.

**Why Track C:** premature for a 1-day-old single-author repo. Defer.

---

### F13 — CONTRIBUTING.md absent

**Severity:** IMPORTANT · **Track:** C · **Lens:** Production readiness ·
**Status:** `[open]` · **Contributes to thesis debt:** none

**Evidence:**

- No `CONTRIBUTING.md` at repo root.
- post_transformers added it with F12 (same commit `ebdf96b`).

**Recommendation (deferred to Track C):**

- Land with F12. Use
  [`post_transformers/CONTRIBUTING.md`](https://github.com/brandon-behring/post_transformers/blob/main/CONTRIBUTING.md)
  as template, adapted for book context (editorial conventions for prose
  contributions, companion-code contributions, no PRs without prior issue
  discussion).

**Why Track C:** premature without external contributors.

---

### F14 — No CI workflow

**Severity:** IMPORTANT · **Track:** C · **Lens:** Production readiness ·
**Status:** `[open]` · **Contributes to thesis debt:** none

**Evidence:**

- No `.github/workflows/` directory.
- Cloudflare Workers Builds auto-deploys on push to `main` (per CLAUDE.md
  §"Build + deploy"), which gives one form of CI (build-or-die), but no
  validation runs before deploy.

**Recommendation (deferred to Track C):**

- After F9 (pre-commit) is proven, add
  `.github/workflows/validate.yml` running on push and pull_request:
  - `npm ci`
  - `npm run build` (this already chains `predev` → `build:bib` +
    `build:labels` + `validate` and then `astro build && pagefind --site dist`)
  - Once F11 lands: `make status-check` (fails if `STATUS.md` stale).
- Branch protection rule on `main` requiring this workflow to pass.

**Why Track C:** premature until pre-commit + scripts/ + STATUS.md
infrastructure exists; otherwise CI has nothing meaningful to validate.

---

### F15 — No GitHub issue/PR templates

**Severity:** IMPORTANT · **Track:** C · **Lens:** Production readiness ·
**Status:** `[open]` · **Contributes to thesis debt:** none

**Evidence:**

- No `.github/ISSUE_TEMPLATE/` or `.github/pull_request_template.md`.
- CLAUDE.md §"Issue filing" documents the policy (toolkit issues to
  book-scaffold-astro with label `consumer:ssm-foundations`; content issues
  here) but no templates encode it.

**Recommendation (deferred to Track C):**

- Land with F13/F14. Templates:
  - `.github/ISSUE_TEMPLATE/content_issue.yml` — fields: chapter, severity
    (typo / math error / structural), evidence.
  - `.github/ISSUE_TEMPLATE/toolkit_issue.yml` — pre-filled with hint that
    book-scaffold-astro issues should go to that repo with
    `consumer:ssm-foundations`.
  - `.github/pull_request_template.md` — checklist: chapter status updated,
    bib added, figures referenced, exercises checked.

**Why Track C:** premature without external contributors.

---

### F16 — No `.editorconfig`

**Severity:** MINOR · **Track:** C · **Lens:** Authoring velocity ·
**Status:** `[open]` · **Contributes to thesis debt:** Authoring-convention
debt (§2.1, minor)

**Evidence:**

- No `.editorconfig` at repo root.
- Mixed file types: MDX (2-space indent typical), Python (4-space, black
  enforced), Julia (4-space SciML), JSON / YAML (2-space). Risk: single
  editor inconsistency producing mixed-indent files.

**Recommendation (deferred to Track C):**

- Minimal `.editorconfig`:
  ```ini
  root = true

  [*]
  charset = utf-8
  end_of_line = lf
  insert_final_newline = true
  trim_trailing_whitespace = true

  [*.{mdx,md,json,yml,yaml}]
  indent_style = space
  indent_size = 2

  [*.{py,jl}]
  indent_style = space
  indent_size = 4
  ```

**Why Track C:** low-impact polish; black + JuliaFormatter (F8) cover the
actual rigor.

---

### F17 — Practice tags MDX equivalent (pilot-blocked)

**Severity:** IMPORTANT · **Track:** C · **Lens:** Pilot integration ·
**Status:** `[pilot-blocked]` · **Contributes to thesis debt:** none

**Evidence:**

- post_transformers/guides/STANDARDS.md defines practice tags
  `\tagofficial{}` (official recommendation), `\tagpractitioner{}`
  (practitioner consensus), `\tagconv{}` (convention/heuristic) for
  marking actionable recommendations.
- ssm-foundations has no MDX equivalent (would be e.g. `<Tag
  type="official">`).
- Pilot relevance: C1 + B pilot findings (2026-Q3) will produce
  recommendations that need tag taxonomy — e.g., "Use symplectic Euler for
  Hamiltonian-structured Mamba blocks" might be `tagofficial` if the
  evidence is strong, `tagconv` if heuristic.

**Recommendation (deferred, pilot-blocked):**

- **Unblock condition:** C1 first experiments produce at least 3 actionable
  recommendations classifiable into tag categories (target: 2026-Q3).
- When unblocked: create `<Tag type="official|practitioner|conv">` MDX
  component in `src/components/`. Document in STYLE.md (F5).

**Why pilot-blocked:** premature to design tag taxonomy without empirical
claims to tag.

---

### F19 — Ch 6 RK4-vs-symplectic energy drift magnitude claims

**Severity:** IMPORTANT · **Track:** A · **Lens:** Authoring velocity ·
**Status:** `[fixed]` (umbrella issue #1) · **Contributes to thesis debt:** Truthfulness debt (§2.3),
Authoring-convention debt (§2.1)

**Evidence:** Surfaced during F8 implementation. Six contradictions across
`src/content/chapters/ch06-implicit-and-symplectic.mdx:111,137,155,198` and
`companions/ch06/julia/symplectic_methods.jl:184-186`. At the chapter's cited
parameters (harmonic oscillator, $\Delta = 0.05$, 100 periods), RK4 energy
drift is $1.36 \times 10^{-6}$ — three orders smaller than the figure
caption's "$\sim 10^{-3}$" claim and four orders smaller than Symplectic
Euler's bounded-oscillation band ($2.5 \times 10^{-2}$). Inspection of the
actual figure files (`public/figures/ch06/energy_drift.png`,
`phase_portrait.png`) further revealed (a) `energy_drift.png` is plotted
at $\Delta = 0.3$, not $\Delta = 0.05$ as the caption claimed, and (b) the
phase portrait at $\Delta = 0.1 / 50$ periods shows both RK4 and Verlet
trajectories visually indistinguishable (no visible spiraling, contra the
caption text).

**Recommendation (applied):** Fix A — update chapter prose, the Exercise 6.3
solution, and the Julia companion's interpretation block to match
empirical numbers and the actual figure parameters. Reframe the pedagogy
around horizon-invariance of the symplectic band (the truthful qualitative
property), rather than absolute-magnitude dominance at any specific
$(\Delta, T)$.

**Remediation track:** A. Prose + companion text edits only (no figure
regeneration in this commit; figures themselves are honestly described
post-correction). A future commit may regenerate the figures at a $(\Delta, T)$
regime where the qualitative contrast is visually dramatic — that work is
out of scope for this Track A truthfulness fix.

---

### F18 — `precision.md` pattern adoption (pilot-blocked)

**Severity:** IMPORTANT · **Track:** C · **Lens:** Pilot integration ·
**Status:** `[pilot-blocked]` · **Contributes to thesis debt:** none

**Evidence:**

- `/home/brandon_behring/Claude/lever_of_archimedes/patterns/precision.md`
  exists in the patterns hub but is not referenced from
  ssm-foundations/CLAUDE.md.
- post_transformers/CLAUDE.md does not cite `precision.md` either — it's a
  newer pattern (no commit history evidence of explicit adoption upstream).
- Pilot relevance: B (two-timescale benchmarks) will produce numerical
  precision claims (e.g., "f32 vs f64 affects long-horizon energy drift by
  X%"). Without consistent precision-claim framing, results may be reported
  inconsistently across Ch 14 + Ch 16.

**Recommendation (deferred, pilot-blocked):**

- **Unblock condition:** B pilot first results expose precision-sensitivity
  measurements requiring consistent rigor framing (target: 2026-Q3).
- When unblocked: cite `precision.md` in CLAUDE.md hub-reference block (F2)
  and codify any precision conventions in STYLE.md (F5).

**Why pilot-blocked:** pattern is not yet known to be load-bearing for this
repo's content.

---

## 5. Asymmetries that should NOT transfer from post_transformers

These patterns from post_transformers are deliberately *not* adopted by
ssm-foundations. Documenting them prevents future audits from cargo-culting
them in.

| post_transformers pattern | Why not transferred |
|---|---|
| Cloud GPU / RunPod orchestration (`configs/runpod/`, `make smoke-cloud`, `make bench-cloud`) | ssm-foundations is a teaching book; companion code is illustrative, not large-scale. No GPU workload here. |
| `references/dossier/` + `references/paper_index.md` heavy research apparatus | Book uses `bibliography.bib`-driven workflow appropriate to a public book. Research-dossier maintenance is post_transformers' job. |
| `weekNN/` code-first companion structure | ssm-foundations is correctly chapter-first (`chXX/`) via book-scaffold-astro inline-companions. |
| pytest markers `slow` / `gpu` / `pytorch` | Over-engineered for book companions. Companion code should run quickly on a laptop. |
| Emoji status markers in audit tables (the four glyphs used in post_transformers' April 11 audit Status column for monitoring / tracked / fixed / blocked) | Replaced with bracketed text labels — see §3 Status vocabulary. Future audits in this repo follow text-label convention for grep/log durability. |
| `paper_index.md` + `wish_list.md` + `evidence_matrix.md` triad | Book bibliography is leaner (16 entries vs ~98); the elaborate ledger system is research-team infrastructure, not book infrastructure. |

This section prevents accidentally over-importing standards that don't fit
the book context.

---

## 6. Verification appendix

### post_transformers artifacts referenced

All URLs are absolute GitHub URLs pinned to `main`, per CLAUDE.md
§"Cross-repo link convention":

| Reference | Used in finding(s) |
|---|---|
| [`audits/archive/2026-04-11_repo_maintainability_audit.md`](https://github.com/brandon-behring/post_transformers/blob/main/audits/archive/2026-04-11_repo_maintainability_audit.md) | Format template (§3 table structure, Track A/B/C) |
| [`audits/README.md`](https://github.com/brandon-behring/post_transformers/blob/main/audits/README.md) | F3 (audits/ pattern) |
| [`CLAUDE.md`](https://github.com/brandon-behring/post_transformers/blob/main/CLAUDE.md) | F2 (hub reference block at L114–121) |
| [`CONTRIBUTING.md`](https://github.com/brandon-behring/post_transformers/blob/main/CONTRIBUTING.md) | F12, F13 (license split language) |
| [`.pre-commit-config.yaml`](https://github.com/brandon-behring/post_transformers/blob/main/.pre-commit-config.yaml) | F9 (Makefile-routed pattern) |
| [`Makefile`](https://github.com/brandon-behring/post_transformers/blob/main/Makefile) | F10 (gate sequencing) |
| [`docs/STATUS.md`](https://github.com/brandon-behring/post_transformers/blob/main/docs/STATUS.md) | F11 (status snapshot template) |
| [`guides/STANDARDS.md`](https://github.com/brandon-behring/post_transformers/blob/main/guides/STANDARDS.md) | F5 (style-guide template), F17 (practice tags) |
| [`notes/niche_decision_2026_05_24.md`](https://github.com/brandon-behring/post_transformers/blob/main/notes/niche_decision_2026_05_24.md) | F8 (C1 pilot timing) |

### ssm-foundations file paths referenced in findings

| Path | Exists? | Referenced in |
|---|---|---|
| `audits/README.md` | YES (created by this audit) | F3 |
| `audits/2026-05-25_standards_vs_post_transformers.md` | YES (this file) | F3 |
| `CLAUDE.md` | YES; edits proposed | F1, F2 |
| `companions/README.md` | YES; edits proposed | F1, F4 |
| `companions/_shared/plot_utils.py` | YES; cited at L1–16 | F4 |
| `companions/_shared/pyproject.toml` | YES; cited at L26, L34, L39 | F8 |
| `companions/_shared/JuliaFormatter.toml` | NO; create | F8 |
| `companions/ch{04,05,06}/julia/runtests.jl` | NO (×3); create | F8 |
| `companions/ch{01,02,03,04,05,06}-*.mdx` (chapter content) | YES (six files) | F1 |
| `bibliography.bib` | YES (16 entries) | F6 |
| `scripts/check-bibkeys.mjs` | NO; create | F6 |
| `scripts/check-xref-labels.mjs` | NO; create | F7 |
| `scripts/generate-status.mjs` | NO; create | F11 |
| `.pre-commit-config.yaml` | NO; create | F9 |
| `Makefile` | NO; create | F10 |
| `docs/STATUS.md` | NO; create | F11 |
| `LICENSE` | NO; deferred | F12 |
| `LICENSE-CONTENT` | NO; deferred | F12 |
| `CONTRIBUTING.md` | NO; deferred | F13 |
| `.github/workflows/validate.yml` | NO; deferred | F14 |
| `.github/ISSUE_TEMPLATE/` | NO; deferred | F15 |
| `.github/pull_request_template.md` | NO; deferred | F15 |
| `.editorconfig` | NO; deferred | F16 |
| `STYLE.md` (or `AUTHORING.md`) | NO; create | F5 |

---

## 7. Next steps

After this audit lands:

1. **File the umbrella tracking issue** at
   `brandon-behring/ssm-foundations` (title: "Standards hardening per
   2026-05-25 audit") with a checklist body of F1–F18, each linking to
   the relevant `#f1` … `#f18` anchor in this document. Labels: `tracked`,
   `improvement`. Per-finding sub-issues created later via `gh sub_issue_write`
   when each finding is picked up.
2. **Schedule Track A remediation session** within the next 7 days,
   before the C1 pilot lands on 2026-06-01. Priority order: F8 (rigor must
   precede pilot code), F1 (truthfulness in onboarding), F4 (port-credit
   convention before more ports), F2 (hub block), F3 (already fixed).
3. **Track B work** is GH-issue tracked and picked up incrementally over
   2026-Q3. Recommended order: F5 (style guide) → F11 (STATUS.md) → F10
   (Makefile) → F6 + F7 (lint scripts) → F9 (pre-commit).
4. **Track C work** waits for the appropriate trigger: F12–F15 await the
   first external contributor / pre-1.0 milestone; F17 + F18 await pilot
   first results in 2026-Q3.
5. **Schedule the next canonical audit** for ~2026-07-25 (8 weeks out,
   matching post_transformers' early audit cadence). Expected questions
   it will answer: "Did Track A remediation hold? Are Track B items
   tracking? Are pilot-blocked findings now answerable?"
