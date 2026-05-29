# ssm-foundations — AI authoring guide

A 17-chapter lens-led foundations book for sequence-model architectures, with the dynamical-systems perspective foregrounded. Built with `@brandon_m_behring/book-scaffold-astro` (academic preset, v4.8.0).

**Status**: scaffold landed 2026-05-24 (Plan 3 launch). Chapters 1–6 are authored (`status: implemented`); Ch 7–17 are stubbed `planned`.

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

Current state (2026-05-25): Ch 1–6 = `implemented`; Ch 7–17 = `planned`.

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
- v3→v4 migration: [`MIGRATION-v3-to-v4.md`](https://github.com/brandon-behring/book-scaffold-astro/blob/main/package/MIGRATION-v3-to-v4.md). (Not needed for this book — scaffolded fresh on v4.2.0, now tracking v4.8.0.)
