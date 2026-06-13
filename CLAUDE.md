# ssm-foundations — AI authoring guide

A 17-chapter lens-led foundations book for sequence-model architectures, with the dynamical-systems perspective foregrounded. Built with `@brandon_m_behring/book-scaffold-astro` (academic preset, v4.16.0).

**Status**: scaffold landed 2026-05-24 (Plan 3 launch). Chapters 1–14 and 16 are authored (`status: implemented`) — foundations (1–6) + SSM-core (7–10) + beyond-SSM through the delta-rule lineage and the exponential-gate/matrix-memory architectures (11–13) + integration through hybrids and empirical methodology (14, 16 — the pilot-B anchors; **B's book-side prerequisites closed at Ch 16's merge**) shipped and deployed; Ch 15 and 17 are stubbed `planned`. (Status line updated 2026-06-13.)

---

## Lineage and cross-repo context

- **Curriculum design**: [`post_transformers/notes/foundations_curriculum_design_2026_05_20.md`](https://github.com/brandon-behring/post_transformers/blob/main/notes/foundations_curriculum_design_2026_05_20.md) — the 17-chapter design and per-chapter scope.
- **Crosswalk**: foundations §22 — bidirectional mapping between this book's chapters and the predecessor `post_transformers/roadmap.md` W1–W21 weeks.
- **Niche commitment**: [`post_transformers/notes/niche_decision_2026_05_24.md`](https://github.com/brandon-behring/post_transformers/blob/main/notes/niche_decision_2026_05_24.md) — C1 (symplectic integrators) primary + B (two-timescale benchmarks) parallel. The C1 pilot specifically anchors on Ch 1–3 + Ch 6.
- **Predecessor curriculum**: [`post_transformers/roadmap.md`](https://github.com/brandon-behring/post_transformers/blob/main/roadmap.md) — 21-week curriculum, stays canonical inside post_transformers; this book is the lens-led reorganization, not a replacement.

---

## Where things live

- **Chapters**: `src/content/chapters/chXX-slug.mdx`. Frontmatter follows the academic schema. Filename uses `chXX-slug.mdx`; the schema field `week: N` carries the integer ordering (read as "chapter N" for this book).
- **Companions**: `companions/chXX/{jax,julia,torch}/` — per-chapter, per-language code companions. Inline-companion routing wired via book-scaffold-astro 3.2's inline-companions feature.
- **Exercises**: embedded in each chapter MDX as a `## X.N Exercises` section (numbered problems with inline `<details>` solutions for short/numerical, plus a `## X.N+1 Full solutions` section for theory exercises). Not separate files.
- **Bibliography**: `bibliography.bib` → `src/data/references.json` via `npm run build:bib`.
- **Cross-references**: `id="..."` on `<Theorem>` / `<Figure>` → `src/data/labels.json` via `npm run build:labels`.

---

## Authoring standards (STYLE.md at a glance)

`STYLE.md` is the authoritative authoring contract; this index tells you what to read before drafting. §11 (the Ch 7–17 checklist) is the spine the `chapter-auditor` enforces; §13 lists the sanctioned Ch 1 / Ch 5 deviations.

| § | Topic | § | Topic |
|---|---|---|---|
| 1 | Chapter skeleton (frontmatter + positional sections) | 8 | Companion code (`## X.10`) |
| 2 | Math notation (KaTeX macros, declared in frontmatter) | 9 | Margin notes (~3, ≤80 words, nothing load-bearing) |
| 3 | Components vocabulary | 10 | Voice and pedagogy (rigor-first, specialist audience) |
| 4 | Theorem cross-references (`id=` / `<XRef>`) | 11 | Authoring checklist for Ch 7–17 |
| 5 | Citations (`<Cite>` + bibkey format) | 12 | Drift and updates |
| 6 | Figures (caption credits the producer script) | 13 | Known exceptions (Ch 1, Ch 5) |
| 7 | Exercises (3 short inline + 3 long) | | |

---

## Memory & session continuity

Session state lives in two places — read both when resuming authoring:

- **`CURRENT_WORK.md`** (repo root) — a one-screen "right now / why / next step" snapshot; the fastest resume context.
- **Claude Code project-memory** — the `MEMORY.md` index under `~/.claude/projects/.../memory/` links the Ch 7–17 roadmap, the chapter-authoring playbook, per-chapter pre-recon, and feedback/convention memories. (Auto-injected at session start; named here so it's discoverable, and so its existence is not invisible to a human reader.)

---

## Status taxonomy (7-state)

Migrated from `post_transformers/roadmap.md`:

| Status | Meaning |
|---|---|
| `implemented` | Prose + exercises + companions all authored, audited, and CI-green |
| `chapter_only` | Prose authored; companions/exercises deferred |
| `reading_only` | Reading checklist / pointer-only; no original prose |
| `prose_only` | Prose authored, companions intentionally omitted |
| `code_only` | Companions authored, chapter still WIP |
| `scaffolded` | Skeleton file with frontmatter + outline, no real content |
| `planned` | No file yet, or a stub with PreReleaseBanner only |

Current state (2026-06-13): Ch 1–14, 16 = `implemented`; Ch 15, 17 = `planned`.

---

## Project subagents (`.claude/agents/`)

Five **read-only review/verify** subagents isolate high-volume, well-specified
checks from the main authoring thread. They auto-fire (precise `description:`
fields + a `PostToolUse` nudge hook in `.claude/settings.json`); all are
findings-only and never edit files. Full index + invocation in
[`.claude/README.md`](.claude/README.md).

| Agent | Fires when | Does |
|---|---|---|
| `chapter-auditor` | a chapter `.mdx` is drafted/edited, before advancing `status:` | STYLE.md + `status:`-truthfulness audit; runs `make lint` / `make validate` |
| `companion-verifier` | companion code under `companions/chXX/**` changes | runs jax/julia/torch suites, JAX↔Julia parity, figure/caption checks |
| `prose-pedagogy-reviewer` | a prose draft reads complete | teaching-quality review vs STYLE.md §§9–10 + Ch 1–6 |
| `citation-link-auditor` | `bibliography.bib` changes / pre-release | bibkey hygiene + `<Cite>` resolution + cross-repo URL freshness |
| `claim-skeptic` | a chapter's math is drafted/edited, before advancing `status:` | adversarial **claim-truth** check: theorem/derivation soundness, attributions, numeric-claim↔companion parity, overclaimed generality |

Authoring stays in the main thread by design (review/verify only); escalate to a
multi-agent **Workflow** if Ch 11–17 needs orchestrated fan-out.

---

## Cross-repo link convention

When referencing back to `post_transformers/` (dossier folders, kickoffs, watchlist), use **absolute GitHub URLs pinned to `main`**:

```
https://github.com/brandon-behring/post_transformers/blob/main/notes/niche_decision_2026_05_24.md
```

Accepted rot risk: paths may rename in post_transformers. Don't use git-submodule, don't use SHA-pinned permalinks. Simpler is better for a public book.

---

## Pilot integration policy

Ch 1–3 + Ch 6 should be authored with the C1 pilot's empirical needs as primary use case. Ch 14 + Ch 16 should be authored with the B pilot's two-timescale benchmark needs as primary use case. Pilot-driven authoring is a feature, not a constraint — the curriculum should be *informed by* what pilots actually need.

---

## Build + deploy

```bash
npm install
npm run dev               # localhost:4321
npm run validate          # content checks (prevalidate hook auto-runs build:bib + build:labels first)
npm run build             # dist/
npx wrangler deploy       # manual one-off (normally CI handles this)
```

**Deploy paradigm**: Cloudflare Workers + Static Assets, deployed via GitHub Actions calling the reusable workflow at [`brandon-behring/deploy-workflows`](https://github.com/brandon-behring/deploy-workflows) (pinned to `@v1`). `.github/workflows/deploy.yml` is a 10-line caller; pushes to `main` trigger production deploys. No Cloudflare Workers Builds (dashboard auto-build) involved — the Actions-owned pipeline is canonical.

- **Worker name**: `brandon-behring-ssm-foundations` (person-prefixed flat per the brandon-behring.dev convention).
- **Production URL**: <https://ssm-foundations.brandon-behring.dev> (custom domain bound in CF dashboard).
- **Workers.dev preview URL**: <https://brandon-behring-ssm-foundations.brandon-m-behring.workers.dev>.
- **First-deploy audit**: [`audits/2026-05-26_first-deploy.md`](audits/2026-05-26_first-deploy.md) — full Phase 1c decisions matrix + findings (including the latent `references.json` gap that drove the `ci:validate` script convention, since superseded by the `prevalidate` hook).

**Deployment URL convention**: each book/project under `brandon-behring.dev` follows the per-project-subdomain pattern. See [the Subdomain convention in brandon-behring.dev/README.md](https://github.com/brandon-behring/brandon-behring.dev#subdomain-convention) for the slug rule, click-path, and registry.

---

## Issue filing

File toolkit issues at https://github.com/brandon-behring/book-scaffold-astro/issues with label `consumer:ssm-foundations` (per the consumer-feedback-driven evolution policy).

File book-content issues at https://github.com/brandon-behring/ssm-foundations/issues.

---

## Hub Pattern References

This book lives in the personal hub-and-spoke architecture rooted at `~/Claude/lever_of_archimedes/patterns/`. The patterns referenced here apply to authoring + tooling decisions in this repo:

- Git commits: `~/Claude/lever_of_archimedes/patterns/git.md` (conventional-commits format, session-based commit cadence).
- Sessions: `~/Claude/lever_of_archimedes/patterns/sessions.md` (multi-sprint workflow).
- Deploy: `~/Claude/lever_of_archimedes/patterns/deploy_subdomain_brandon_behring_dev.md` (subdomain convention — this book deploys at [`ssm-foundations.brandon-behring.dev`](https://ssm-foundations.brandon-behring.dev)).

Substantive adoption of `testing.md` is tracked via [audit F8](audits/2026-05-25_standards_vs_post_transformers.md) (Julia + torch companion rigor).

---

## Toolkit reference

- v4 API: see [PACKAGE_DESIGN.md](https://github.com/brandon-behring/book-scaffold-astro/blob/main/PACKAGE_DESIGN.md) for the canonical contract.
- v3→v4 migration: [`MIGRATION-v3-to-v4.md`](https://github.com/brandon-behring/book-scaffold-astro/blob/main/package/MIGRATION-v3-to-v4.md). (Not needed for this book — scaffolded fresh on v4.2.0, now tracking v4.16.0.)
