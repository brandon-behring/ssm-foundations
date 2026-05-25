# ssm-foundations — AI authoring guide

A 17-chapter lens-led foundations book for sequence-model architectures, with the dynamical-systems perspective foregrounded. Built with `@brandon_m_behring/book-scaffold-astro` (academic preset, v4.2.0).

**Status**: scaffold landed 2026-05-24 (Plan 3 launch). Chapters 1–3 are next-up for authoring; Ch 4–17 are stubbed `planned`.

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
- **Exercises**: `src/content/chapters/chXX/exercises.mdx` (separate file per chapter; gated by `<BlockedByCallout>` until authored).
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

Current state (2026-05-24): Ch 1–3 = `scaffolded` (authoring imminent); Ch 4–17 = `planned`.

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
npm run validate          # content checks
npm run build             # dist/
npx wrangler deploy       # Cloudflare Workers + Static Assets
```

**Deploy target**: Cloudflare Workers (toolkit default for academic preset). Public site will land at `ssm-foundations.<account>.workers.dev` until a custom domain is wired.

---

## Issue filing

File toolkit issues at https://github.com/brandon-behring/book-scaffold-astro/issues with label `consumer:ssm-foundations` (per the consumer-feedback-driven evolution policy).

File book-content issues at https://github.com/brandon-behring/ssm-foundations/issues.

---

## Toolkit reference

- v4 API: see [PACKAGE_DESIGN.md](https://github.com/brandon-behring/book-scaffold-astro/blob/main/PACKAGE_DESIGN.md) for the canonical contract.
- v3→v4 migration: [`MIGRATION-v3-to-v4.md`](https://github.com/brandon-behring/book-scaffold-astro/blob/main/package/MIGRATION-v3-to-v4.md). (Not needed for this book — scaffolded fresh on v4.2.0.)
