# Contributing

This is a personal research and authoring repository for a 17-chapter
lens-led foundations book on post-transformer sequence-model architectures.
Issues and pull requests are welcome but expectations should be calibrated.

- **Typos, broken links, factual errors, magnitude-checking fact-checks
  on empirical claims**: please open an issue or send a small PR. These
  are the easiest contributions to accept. The book's authoring conventions
  (see [`STYLE.md`](STYLE.md)) cover the expected format.
- **Larger structural suggestions** (new chapter ordering, alternative
  proofs, missing references, additional exercises): an issue first to
  discuss scope. The 17-chapter design is load-bearing; structural
  changes need alignment with the
  [curriculum design doc](https://github.com/brandon-behring/post_transformers/blob/main/notes/foundations_curriculum_design_2026_05_20.md)
  in the sibling research repo.
- **New chapters or substantial prose**: probably best as a fork. The
  voice and pedagogy are deliberate (and personal); building on top of
  the source is usually a cleaner path than folding contributions back.

## License

This repository uses a code/content license split — see also
[`LICENSE`](LICENSE) and [`LICENSE-CONTENT`](LICENSE-CONTENT).

- **Code** ([`companions/`](companions/), [`scripts/`](scripts/), Astro/
  TypeScript site infrastructure, build configs): MIT — see
  [`LICENSE`](LICENSE).
- **Prose** (chapter MDX, [`audits/`](audits/), [`STYLE.md`](STYLE.md),
  README, etc.): CC BY 4.0 — see [`LICENSE-CONTENT`](LICENSE-CONTENT).

Submitting a PR means you agree to license your contribution under those
terms (MIT for code changes, CC BY 4.0 for prose changes).

## Local development

```bash
npm install
npm run dev               # localhost:4321, hot reload
npm run build             # production build to dist/
make check                # full gate: validate + bibkey lint + xref lint + status-check
```

Optional, for companion code:

```bash
# Python/JAX companions (companions/_shared/pyproject.toml)
cd companions/_shared && pip install -e .[dev]
PYTHONPATH=. python companions/ch01/jax/damped_oscillator.py

# Julia companions (companions/_shared/JuliaFormatter.toml + per-chapter Project.toml)
julia --project=companions/ch06/julia companions/ch06/julia/runtests.jl
```

The repo's pre-commit hooks (see [`.pre-commit-config.yaml`](.pre-commit-config.yaml))
run automatically via the global gitleaks pre-commit wrapper when both
`.pre-commit-config.yaml` exists and the `pre-commit` tool is installed:

```bash
uv tool install pre-commit   # one-time, user-global, not a project dep
```

## Authoring a new chapter

See [`STYLE.md`](STYLE.md) for the canonical conventions (frontmatter
schema, KaTeX macros, component vocabulary, theorem cross-ref IDs,
bibkey format, figure paths, exercise structure, companion code
references, margin notes, voice). The authoring checklist in
[STYLE.md §11](STYLE.md) is the pre-commit mental gate.

## What's in the repo

- [`src/content/chapters/`](src/content/chapters/) — 17 chapter MDX
  files (Ch 1–6 implemented, Ch 7–17 planned as of the latest
  [STATUS.md snapshot](docs/STATUS.md))
- [`companions/`](companions/) — per-chapter companion code in
  JAX/Julia/torch (see [companions/README.md](companions/README.md))
- [`audits/`](audits/) — audit reports (canonical:
  [`2026-05-25_standards_vs_post_transformers.md`](audits/2026-05-25_standards_vs_post_transformers.md))
- [`bibliography.bib`](bibliography.bib) — single source of truth for
  citations; regenerated to `src/data/references.json` via `npm run build:bib`
- [`STYLE.md`](STYLE.md) — authoring conventions
- [`CLAUDE.md`](CLAUDE.md) — onboarding guide

## Filing issues

Per the [work-tracking convention](https://github.com/brandon-behring/post_transformers/blob/main/CONTRIBUTING.md)
in the sibling repo:

- **Toolkit (book-scaffold-astro) issues**: file at
  [book-scaffold-astro](https://github.com/brandon-behring/book-scaffold-astro/issues)
  with label `consumer:ssm-foundations`.
- **Book content issues**: file at
  [ssm-foundations](https://github.com/brandon-behring/ssm-foundations/issues)
  using the issue templates under `.github/ISSUE_TEMPLATE/`.

## Cross-repo references

The companion research repo
[`post_transformers`](https://github.com/brandon-behring/post_transformers)
contains the 21-week research curriculum that this book is the lens-led
reorganization of. Links to it use absolute GitHub URLs pinned to
`main`, per the convention documented in [`CLAUDE.md`](CLAUDE.md) §"Cross-repo
link convention".
