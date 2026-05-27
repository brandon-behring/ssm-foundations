# ssm-foundations - Repo Content, Correctness, Code Quality, and Organization Audit

**Audit date:** 2026-05-27
**Scope:** comprehensive repo audit, not a full formal proof audit of every derivation
**Artifact policy:** findings-only; this audit intentionally does not remediate findings
**Compared against:** local sibling `../post_transformers`, especially notebook policy and standards

## Executive Summary

The repo is in a much stronger infrastructure state than the inaugural
2026-05-25 audit: the core content gates pass, the production build works, npm
audit is clean, CI/deploy scaffolding exists, and the Ch 1-6 companion-code
surface is real rather than decorative.

The main risks are now truthfulness and reproducibility gaps at the edges of
that stronger baseline:

1. Public-facing docs still contain stale status/version claims.
2. The repo has no notebooks, unlike `post_transformers`, and has not yet made
   the replacement policy explicit.
3. Figure cross-reference discipline is internally inconsistent: `STYLE.md`
   and `CLAUDE.md` treat figures as labelable, but every actual `<Figure>` has
   no `id`, and the lint skips missing IDs.
4. Ch 5 contains a likely Runge-Kutta order-condition count error.
5. Mamba-3 claims are current and source-checkable, but the repo lacks a
   Mamba-3 bibliography entry and some prose makes uncited claims about that
   paper's discretization.

Positive baseline:

- `make check` passes: academic validation, bibkey lint, xref lint, and status
  staleness.
- `npm run build` passes and builds 22 pages plus Pagefind.
- `npm audit --audit-level=moderate` reports 0 vulnerabilities.
- Python companions compile with `python3`.
- Julia companion tests for Ch 5 and Ch 6 pass.

## Severity-Ranked Findings

| ID | Severity | Area | Finding | Priority |
|---|---|---|---|---|
| A1 | High | Truthfulness | `README.md` still says Ch 1-3 are active and Ch 4-17 are stubbed; current repo state is Ch 1-6 implemented, Ch 7-17 planned. | Fix next |
| A2 | High | Content correctness | Ch 5 likely undercounts general RK order conditions and says RK4 has "all 4 order-4 conditions"; rooted-tree evidence indicates 8 cumulative conditions through order 4. | Fix next |
| A3 | High | Cross-references | All 15 chapter `<Figure>` blocks lack `id`, while repo docs say figures feed labels; `check-xref-labels` skips missing IDs. | Fix next |
| A4 | Important | Notebook parity | `ssm-foundations` has no notebooks; `post_transformers` has 42 guide notebooks and explicit notebook gates. This needs a policy decision. | Decide soon |
| A5 | Important | Current research claims | Mamba-3 claims are source-checkable, but there is no Mamba-3 bib entry and some claims are uncited or stronger than the paper text inspected. | Fix with Ch 10 work |
| A6 | Important | Reproducibility | Ch 4 Julia test is real but does not run on a fresh local environment until `Pkg.instantiate()` is done; default Makefile excludes it. | Improve gate/docs |
| A7 | Important | Docs/tooling drift | Docs show `book-scaffold-astro` v4.2.0 while package files use v4.5.1; docs also use bare `python`, unavailable in this shell. | Fix soon |
| A8 | Moderate | Asset hygiene | Two committed figure assets are unused by chapter MDX. | Triage |
| A9 | Moderate | Dependency freshness | `npm outdated` shows Astro patch drift and KaTeX 0.17.0 available. No vulnerability, but record dependency drift. | Monitor |
| A10 | Moderate | Organization docs | `CLAUDE.md` says exercises live in separate per-chapter files, but actual Ch 1-6 exercises are embedded in chapter MDX. | Fix docs |

## Per-Finding Detail

### A1 - README Status Is Stale

**Evidence**

- `README.md:5` says: Ch 1-3 are in active authoring and Ch 4-17 are stubbed.
- `docs/STATUS.md` and chapter frontmatter show Ch 1-6 are `implemented` and
  Ch 7-17 are `planned`.
- `README.md:19` already describes "Foundations only (Ch 1-6)", contradicting
  the status line above it.

**Impact**

The README is the first public truth surface. A reader will underestimate the
implemented content and misunderstand what is actually ready.

**Recommended fix**

Update the README status block to match `docs/STATUS.md`: Ch 1-6 implemented,
Ch 7-17 planned, production URL live, breaking changes still possible.

### A2 - Ch 5 RK Order-Condition Counts Appear Wrong

**Evidence**

- `src/content/chapters/ch05-stability-regions.mdx:96` says "5 conditions for
  order 4, 17 for order 5, 37 for order 6".
- `src/content/chapters/ch05-stability-regions.mdx:102` says Exercise 5.1
  verifies RK4 satisfies "all 4 order-4 conditions."
- RootedTrees.jl's RK order-condition tutorial verifies classical RK4 against
  8 rooted-tree conditions through order 4 and then checks 9 additional order-5
  trees. Source: <https://sciml.github.io/RootedTrees.jl/stable/tutorials/RK_order_conditions/>.

**Assessment**

The prose appears to mix at least three different counts:

- exact-order rooted trees,
- cumulative conditions through order `p`,
- simplified scalar/autonomous conditions.

For general systems, the standard cumulative sequence through orders 1-6 is
1, 2, 4, 8, 17, 37. The current "5 conditions for order 4" is therefore likely
wrong or at least ambiguous enough to mislead readers.

**Recommended fix**

Rewrite Ch 5 section 5.2 to distinguish scalar simplified conditions from
general-system rooted-tree counts. Update Exercise 5.1 so it says it verifies a
subset of RK4 order conditions, not all order-4 conditions, unless the full set
is included.

### A3 - Figure IDs Are Documented But Not Enforced

**Evidence**

- `CLAUDE.md:24` says cross-references use IDs on `<Theorem>` / `<Figure>`.
- `STYLE.md` section 4 says every `<Theorem>` and `<Figure>` block should use
  `id="ch##:<type>:<slug>"`.
- Current counts:
  - `<Figure>` blocks in chapter MDX: 15
  - `<Figure>` blocks with `id=`: 0
  - committed figure image files: 17
- `scripts/check-xref-labels.mjs:46-47` explicitly continues when no `id` is
  present: missing IDs are "not required (yet)".
- `make check` reports only 14 IDs, matching theorem IDs, not figure IDs.

**Impact**

This makes figure references non-addressable via `XRef` and lets future figure
cross-reference drift pass CI.

**Recommended fix**

Choose one policy and encode it:

- If figures must be referenceable, add `id` to all 15 figures and make
  `check-xref-labels` fail on missing figure IDs.
- If figures are not meant to be referenceable yet, update `STYLE.md` and
  `CLAUDE.md` so they do not claim figure ID parity.

Given the scaffold's `build-labels` support for `Figure`, the recommended
policy is to add figure IDs and enforce them.

### A4 - Notebook Parity With `post_transformers` Is Undecided

**Evidence**

- `find . \( -name '*.ipynb' -o -type d -name 'notebooks' \) -print` returns
  no files or directories in `ssm-foundations`.
- `../post_transformers/guides/notebooks` contains 42 notebooks:
  `week01_notebook.ipynb` through `week21_notebook.ipynb` plus paired
  `weekNN_syntax_notebook.ipynb`.
- `post_transformers/guides/README.md` documents a weekly workflow where each
  week gets a main notebook and syntax supplement.
- `post_transformers/Makefile` has notebook gates: `notebook-check`,
  `nbtest`, and `nbtest-ci`.
- `package.json:13` in this repo exposes `build:notebooks`, but there are no
  source notebooks for it to render.

**Assessment**

This is not automatically a defect. `ssm-foundations` may intentionally use
chapter-embedded exercises plus `companions/chXX/{jax,julia,torch}` scripts as
the notebook replacement. But that policy is not written down, so a reader who
knows `post_transformers` will reasonably ask whether notebooks were forgotten.

**Recommended fix**

Make an explicit policy decision in `README.md`, `STYLE.md`, and
`companions/README.md`:

- Option 1: no notebooks; companion scripts are the executable artifact.
- Option 2: add Ch 1-6 notebooks, likely one main notebook per chapter and no
  syntax supplement unless there is a clear learning need.
- Option 3: notebooks only for pilot-heavy chapters, starting with Ch 6 and
  later Ch 10/14/16.

Recommended default: Option 1 for now, with a revisit trigger when Ch 10 or
Ch 16 lands.

### A5 - Mamba-3 Claims Need Bibliographic Grounding

**Evidence**

- Mamba-3 appears in current chapter prose and planned Ch 10 metadata:
  `src/content/chapters/ch04-discretization.mdx`, `ch05-stability-regions.mdx`,
  `ch06-implicit-and-symplectic.mdx`, and `ch10-mamba-3.mdx`.
- `bibliography.bib` has Mamba-1 and Mamba-2 entries, but no Mamba-3 entry.
- Source check:
  - ICLR lists "Mamba-3: Improved Sequence Modeling using State Space
    Principles" as a 2026 oral:
    <https://iclr.cc/virtual/2026/events/oral>.
  - arXiv entry `2603.15569` confirms the title, authors, ICLR 2026 comment,
    complex-valued state update, and MIMO formulation:
    <https://arxiv.org/abs/2603.15569>.
  - The OpenReview PDF states that Mamba-3 introduces an
    exponential-trapezoidal discretization and describes it as a second-order
    approximation of the state-input integral:
    <https://openreview.net/pdf?id=HwCvaJOiCj>.

**Assessment**

The broad Mamba-3 references are real and current. The issue is citation and
precision: local prose sometimes treats Mamba-3 claims as established chapter
background without a direct `<Cite>` target, and some claims such as A-stability
for the Mamba-3 scheme should be checked carefully against the paper's exact
scope.

**Recommended fix**

Add a Mamba-3 bibliography entry before Ch 10 authoring or any further
Mamba-3-heavy edits. Add direct `<Cite>` references where current prose says
"Mamba-3 switched..." or "Mamba-3 adopts..." Re-audit claims about
A-stability, stiffness, and bilinear/exponential-trapezoidal relationships
against the paper text rather than relying on summary memory.

### A6 - Ch 4 Julia Test Requires Manual Instantiation

**Evidence**

- `make companion-julia-tests` runs Ch 5 and Ch 6 only and passes:
  - Ch 5: 36 tests pass
  - Ch 6 implicit methods: 5 tests pass
  - Ch 6 symplectic methods: 6 tests pass
- Direct Ch 4 command fails:
  `julia --project=companions/ch04/julia companions/ch04/julia/runtests.jl`
- Error: package `DifferentialEquations` is required but not installed; Julia
  recommends running `Pkg.instantiate()`.
- The Makefile comment already documents Ch 4's instantiate requirement.

**Impact**

The Ch 4 test is real but not part of the default companion gate. That is a
reasonable speed trade-off, but it weakens reproducibility for a chapter whose
figures and claims depend on the Julia atlas.

**Recommended fix**

Add a separate `companion-julia-tests-full` target that instantiates and runs
Ch 4, Ch 5, and Ch 6. Keep the current fast target if desired, but document the
split as fast vs full.

### A7 - Docs Drift From Actual Tooling

**Evidence**

- `package.json:23` uses `@brandon_m_behring/book-scaffold-astro` `^4.5.1`.
- `package-lock.json` resolves `book-scaffold-astro-4.5.1`.
- `README.md:48` says v4.2.0.
- `CLAUDE.md:3` and `CLAUDE.md:103` say scaffolded fresh on v4.2.0.
- `STYLE.md:289-290` and `CONTRIBUTING.md:49` use bare `python` commands.
- `python --version` fails in this shell with `command not found`; `python3`
  works and Python companion syntax compilation passes under `python3`.

**Impact**

New contributors or future automation can follow docs and hit avoidable
failures, especially on systems without a `python` shim.

**Recommended fix**

Update docs to say the repo currently uses scaffold v4.5.1 while noting it was
initially scaffolded on v4.2.0 if that history matters. Replace bare `python`
examples with `python3` or a repo-managed environment command.

### A8 - Unused Figure Assets

**Evidence**

All referenced figure sources resolve, but two committed files are not
referenced by any chapter `<Figure src="...">`:

- `public/figures/ch01/matrix_exponential_convergence.png`
- `public/figures/ch06/stiff_blowup.png`

**Impact**

Unused assets are not harmful by themselves, but they create ambiguity: either
chapter prose forgot to include them, or stale generated artifacts are being
carried forward.

**Recommended fix**

For each unused asset, decide whether to add a figure block, mention it only in
companion-code prose, or delete/regenerate it. If it is intentionally
companion-only output, document that convention.

### A9 - Dependency Freshness Is Mostly Clean, With Minor Drift

**Evidence**

- `npm audit --audit-level=moderate`: 0 vulnerabilities.
- `npm outdated --long` reports:
  - `astro`: current 6.3.7, wanted/latest 6.3.8
  - `katex`: current/wanted 0.16.47, latest 0.17.0

**Impact**

No immediate security issue. This is a maintenance watch item.

**Recommended fix**

Patch Astro during the next maintenance batch. Treat KaTeX 0.17.0 as a
compatibility update and test math rendering before adopting.

### A10 - Organization Docs Misdescribe Exercise Layout

**Evidence**

- `CLAUDE.md:22` says exercises live in
  `src/content/chapters/chXX/exercises.mdx`.
- Actual Ch 1-6 exercises are embedded directly in the chapter MDX files.
- `STYLE.md` correctly describes embedded exercise sections.

**Impact**

This is small but confusing for future authoring: the AI guide points to a
nonexistent layout.

**Recommended fix**

Update `CLAUDE.md` to match lived practice: exercises are embedded in the
chapter file under the third-to-last section unless a future chapter explicitly
adopts split exercise files.

## Notebook Parity Section

The answer to "did we make notebooks like we did in post-transformers?" is no.

`ssm-foundations` currently has no `.ipynb` files and no `notebooks/`
directory. It instead has:

- 15 referenced chapter figures under `public/figures/ch01` through `ch06`
- 17 total committed figure files
- 13 JAX/Python files under `companions/`
- 7 Julia/source-test files plus 3 Julia `Project.toml` files
- embedded exercises and worked solutions in Ch 1-6 MDX

`post_transformers` has a different artifact model:

- 42 guide notebooks in `guides/notebooks`
- paired main and syntax notebooks for weeks 1-21
- rendered notebook HTML under the guide web/public area
- Makefile targets for notebook hygiene and execution

Recommendation: do not backfill notebooks by default until the repo states the
policy. The cleanest near-term position is:

> `ssm-foundations` uses companion scripts plus embedded exercises as the
> executable artifact. Notebook backfills are optional and should be introduced
> only for chapters where the learning experience needs an interactive notebook
> rather than a script.

Potential revisit triggers:

- Ch 10 Mamba-3 authoring needs an interactive derivation or scan demo.
- Ch 14/16 empirical-methodology chapters need benchmark notebooks.
- Readers ask for executable walkthroughs rather than script companions.

## Verification Commands And Outcomes

Commands were run from `/home/brandon_behring/Claude/ssm-foundations` on
2026-05-27.

| Command | Outcome |
|---|---|
| `git status --short --branch` | `## main...origin/main`; clean before audit file creation |
| `make check` | Pass; `validate` checked 17 chapters with `profile=academic`; bibkeys and xrefs OK; `docs/STATUS.md` fresh |
| `npm run build` | Pass; 22 pages built; Pagefind indexed 22 pages and 5509 words |
| `npm audit --audit-level=moderate` | Pass; 0 vulnerabilities |
| `npm outdated --long` | Exit 1 with dependency drift: Astro 6.3.7 -> 6.3.8, KaTeX latest 0.17.0 |
| `find companions -name '*.py' -print0 | xargs -0 python3 -m py_compile` | Pass |
| `python --version` | Fails; `/bin/bash: python: command not found` |
| `make companion-julia-tests` | Pass; Ch 5 and Ch 6 Julia tests pass |
| `julia --project=companions/ch04/julia companions/ch04/julia/runtests.jl` | Fails until `Pkg.instantiate()` installs `DifferentialEquations` |
| notebook check in this repo | No `.ipynb` files or `notebooks/` directory |
| `find ../post_transformers/guides/notebooks -maxdepth 1 -name '*.ipynb' | wc -l` | 42 notebooks |
| figure reference check | 15 `<Figure>` blocks, 0 with `id=`, 17 committed figure files, 2 unused figure files |

## Overall Assessment

The repo is healthy enough to continue authoring, but it should not add more
high-claim technical prose without tightening the current truthfulness and
reference gaps. The most efficient next remediation batch is:

1. Fix README/CLAUDE/tool-version/python-command drift.
2. Correct Ch 5 RK order-condition language.
3. Decide and document notebook policy.
4. Add figure IDs or revise the figure-ID policy.
5. Add Mamba-3 bibliography support before Ch 10 becomes substantive.

These are small to moderate changes with high leverage because they prevent the
same drift from compounding as Ch 7-17 are authored.
