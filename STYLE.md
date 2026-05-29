# STYLE.md — authoring conventions for ssm-foundations

This file codifies the authoring conventions already lived in Ch 1–6 so
that Ch 7–17 author without drift. It is the canonical reference; if
this document and lived practice disagree, lived practice in the most
recently authored chapter wins and this document gets updated.

Scope: chapter MDX prose and Astro components consumed from
`@brandon_m_behring/book-scaffold-astro`. Companion code lives by its own
language conventions (`companions/_shared/pyproject.toml` for
Python/JAX; `companions/_shared/JuliaFormatter.toml` for Julia; future
torch parity tracked via audit F8).

Sibling reference: `post_transformers/guides/STANDARDS.md` is the
LaTeX-side equivalent for the research codebase. The two documents
intentionally diverge where MDX and LaTeX express conventions
differently (e.g., margin-note length cap, callout taxonomy).

---

## 1. Chapter skeleton

**File layout.** Each chapter lives at
`src/content/chapters/ch##-<slug>.mdx`. The schema is enforced by
`src/content.config.ts` via `defineBookSchemas()`.

**Frontmatter — five required fields:**

```yaml
---
week: <integer>          # ordering (1..17); read as "chapter N" for this book
part: foundations | ...  # part name; current chapters are part="foundations"
title: '<Title Case>'    # chapter title
status: implemented | chapter_only | reading_only | prose_only | code_only | scaffolded | planned
description: '<one-sentence description, shown in TOC + meta tags>'
---
```

The 7-state status taxonomy is documented in CLAUDE.md
§"Status taxonomy (7-state)"; do not duplicate definitions here.

**Section structure (positional, not numeric).** The canonical structure
is content sections → "What's next" → Exercises → Full solutions to
theory exercises → Companion code, in that order. Numerically:

| Section role | Position |
|---|---|
| Content | §X.1 through §X.(N) where N is 5 or 6 |
| What's next | §X.(N+1) |
| Exercises | §X.(N+2) — third-to-last |
| Full solutions to theory exercises | §X.(N+3) — second-to-last |
| Companion code | §X.(N+4) — last |

For 6 content sections (Ch 1–4, Ch 6 — the canonical N) the numbering
is §X.1–§X.6 content, §X.7 What's next, §X.8 Exercises, §X.9 Full
solutions, §X.10 Companion code. For 5 content sections (Ch 5) the
numbering shifts up: §5.7 Exercises, §5.8 Full solutions, §5.9
Companion code. **Numbers depend on content depth; positions are fixed.**
Target N=6 unless the chapter's content genuinely doesn't subdivide
that way.

**Opening NoteBox.** Every chapter opens with a single `<NoteBox>`
titled "Chapter X — at a glance" containing:

- **Goal**: 1–2 sentences on chapter learning objectives.
- **Reading time**: rough estimate (e.g., "~50 minutes prose; 90+ minutes
  with the JAX and Julia companions").
- **Direct-transfer hook** or **Key insight**: one paragraph naming
  what this chapter contributes to the larger lens, particularly any
  C1 (symplectic) or B (two-timescale) pilot connection.

---

## 2. Math notation

**KaTeX macros.** Each chapter declares the macros it uses in a
frontmatter MDX comment at the top of the file:

```mdx
{/*
  Math notation used in this chapter (auto-loaded via ssmMacros):
    \statevec, \statemat, \inputmat, \outputmat, \stepsize  — system + step
    ...
*/}
```

The `ssmMacros` set is auto-loaded by book-scaffold-astro's academic
preset; the comment is for reader orientation, not for the build. List
only the macros this chapter uses.

**Canonical macro vocabulary** (union across Ch 1–6):

| Macro | Meaning |
|---|---|
| `\statevec` | state vector x |
| `\statemat` | dynamics matrix A |
| `\inputmat` | input matrix B |
| `\outputmat` | output matrix C |
| `\feedmat` | feedthrough matrix D |
| `\stepsize` | discretization step Δ |
| `\discA`, `\discB` | discrete-time dynamics / input matrices |
| `\seqlen` | sequence length L |
| `\stabfn` | stability function R |
| `\stabregion` | stability region S |
| `\hamilton` | Hamiltonian H |
| `\symform` | symplectic 2-form ω |
| `\lyapexp` | Lyapunov exponent λ |
| `\jacobian` | Jacobian J |
| `\spectralradius` | spectral radius ρ |
| `\ddt` | d/dt |
| `\norm{·}`, `\abs{·}`, `\tr`, `\rank`, `\diag` | operators |
| `\R`, `\C`, `\N`, `\Z` | number sets |

New macros: add to `ssmMacros` in book-scaffold-astro's config and
document here. Prefer reusing existing macros over inventing close
variants.

---

## 3. Components vocabulary

Five canonical components from book-scaffold-astro plus one
ssm-foundations-local component (`Tag`):

```mdx
import NoteBox from '@brandon_m_behring/book-scaffold-astro/components/NoteBox.astro';
import Theorem from '@brandon_m_behring/book-scaffold-astro/components/Theorem.astro';
import Cite from '@brandon_m_behring/book-scaffold-astro/components/Cite.astro';
import MarginNote from '@brandon_m_behring/book-scaffold-astro/components/MarginNote.astro';
import Figure from '@brandon_m_behring/book-scaffold-astro/components/Figure.astro';
import Tag from '../../components/Tag.astro';  // ssm-foundations-local
```

| Component | Use |
|---|---|
| `NoteBox` | Chapter-opening "at a glance" box. Not used for in-chapter callouts in Ch 1–6. |
| `Theorem` | Definitions, theorems, propositions, lemmas, examples, remarks. |
| `Cite` | Inline citations to `bibliography.bib`. |
| `MarginNote` | Side commentary, technical asides, pilot connections. |
| `Figure` | Images produced by companion code. |
| `Tag` | Practice-tag inline badge for actionable recommendations (see below). |

`<Theorem type="...">` accepts: `definition`, `theorem`, `proposition`,
`lemma`, `example`, `remark` (mirroring LaTeX amsthm).

### Tag (practice-tag inline badge)

`<Tag type="...">` accepts three semantic values for classifying
actionable recommendations:

| `type` | Meaning |
|---|---|
| `official` | Official recommendation — e.g., paper authors' own stated best practice, or a result with strong empirical support. |
| `practitioner` | Practitioner consensus — community-agreed best practice in the absence of a single authoritative statement. |
| `conv` | Convention or heuristic — a useful default with known exceptions. |

Conceptual lineage:
[`post_transformers/guides/STANDARDS.md` §10](https://github.com/brandon-behring/post_transformers/blob/main/guides/STANDARDS.md)
(`\tagofficial`, `\tagpractitioner`, `\tagconv` in LaTeX). Component
source at [`src/components/Tag.astro`](src/components/Tag.astro). The
component landed as audit F17 infrastructure; substantive use awaits
the C1 + B pilots' first actionable empirical claims (target 2026-Q3).
Chapters before that should not import it.

Other components (`BlockedByCallout`, `Companion`) exist in the
book-scaffold-astro toolkit but are unused in Ch 1–6. Ch 7+ may import
them; if use stabilizes, update this section.

---

## 4. Theorem cross-references

**ID format:** `id="ch##:<type>:<slug>"` on every `<Theorem>` and
`<Figure>` block. The slug is short-kebab-case. `<type>` matches the
`type=` prop (definition → `def`, theorem → `thm`, proposition →
`prop`, etc.) — abbreviated when natural:

```mdx
<Theorem id="ch01:def:matexp" type="definition">...</Theorem>
<Theorem id="ch02:thm:lyap-eig" type="theorem">...</Theorem>
<Theorem id="ch04:bilinear-stability" type="proposition">...</Theorem>
```

When the slug is long enough to disambiguate on its own (e.g.,
`ch06:symplectic-modified-hamiltonian`), the `<type>` prefix may be
elided. Both forms are accepted; future xref-label lint (audit F7)
will accept both.

The build pipeline collects IDs via `npm run build:labels` →
`src/data/labels.json`. References resolve through whatever
cross-reference component book-scaffold-astro provides; check examples
in Ch 1–6.

---

## 5. Citations

**Bibkey format:** `<firstauthor><year><firstword>` (lowercase, no
separators). The first author is the surname; the first word is the
first significant title word (skip "A", "The", "On").

Examples in `bibliography.bib`:

- `hairer1993ordinary` — Hairer–Nørsett–Wanner 1993, "Solving Ordinary…"
- `trefethen1997numerical` — Trefethen–Bau 1997, "Numerical Linear…"
- `gu2024mamba` — Gu–Dao 2024, "Mamba: Linear-Time…"
- `dao2024mamba2` — Dao–Gu 2024, "Mamba 2: Transformers…"
- `anonymous2025lyapunov` — Anonymous 2025, "Lyapunov Stability Analysis…"
  (double-blind submissions use `anonymous<year><firstword>`).

The format is byte-identical with
`post_transformers/guides/shared/references.bib`, so cross-repo
references stay consistent.

**Workflow:**

1. Add a new entry to `bibliography.bib` using the bibkey convention.
2. Run `npm run build:bib` to regenerate `src/data/references.json`.
3. Cite in prose with `<Cite key="<bibkey>" />`.
4. A future lint (audit F6) will validate `<Cite key="">` resolves and
   bibkeys match the format regex.

---

## 6. Figures

**Path:** `src="/figures/ch##/<slug>.png"`. The figure file lives at
`public/figures/ch##/<slug>.png` and is produced by a script under
`companions/ch##/<lang>/`.

**Caption requirements:**

- Describe the **actual figure content** — parameters, system,
  observed magnitudes. (The F19 fix established this — captions that
  describe wished-for figure content rather than actual figure content
  are a truthfulness debt.)
- Cite the producer: "Produced by `companions/ch##/<lang>/<script>.<py|jl>`."
- For figures that demonstrate empirical claims, name the test that
  verifies the claim's order of magnitude. Example: "Verified by
  `companions/ch06/julia/runtests.jl`".

**Alt text** is required, descriptive, and matches the caption's
intent (not a copy of the caption — alt is for screen readers and
should be succinct).

---

## 7. Exercises

**Default count:** 6 per chapter — 3 short (computation or code-run)
plus 3 long (theory or proof). Ch 1 has 7 (1 extra long-form theory
exercise); see §13 for the exception note.

**Short exercises (1 through 3):** stated in §X.(third-to-last) with
inline solutions in `<details><summary>Solution</summary>` blocks
immediately following the exercise statement.

**Long exercises (4 through 6):** stated in §X.(third-to-last) with
"— solution in §X.(second-to-last)" suffix; full worked solutions
live in §X.(second-to-last) under `### Solution to Exercise X.N`
headings.

**Solution conventions:**

- Proofs end with ∎ (the Halmos QED mark) when they conclude.
- Solution prose mirrors the chapter's voice — formal but direct.
- Cross-references to chapter prose use "§X.N" inline.

---

## 8. Companion code

**Reference style: narrative, not component-based.** Chapters do NOT
use a `<Companion>` MDX component. Instead, §X.(last) lists companions
in prose:

```mdx
## X.10 Companion code

Three runnable companions in `companions/chXX/jax/`:

- `<script1>.py` — <one-line description, mentions figures produced>
- `<script2>.py` — <…>
- `<script3>.py` — <…>

To run from the repo root:

```bash
PYTHONPATH=. python companions/chXX/jax/<script1>.py
PYTHONPATH=. python companions/chXX/jax/<script2>.py
```
```

For Julia companions:

```bash
julia --project=companions/chXX/julia companions/chXX/julia/<script>.jl
```

Tests (when present): `julia --project=companions/chXX/julia
companions/chXX/julia/runtests.jl`.

**Port-credit convention.** Any companion file derived from
`post_transformers/` must cite source in its first docstring/comment
block with an absolute GitHub URL pinned to `main`. See
`companions/README.md` §"Port-credit convention" (audit F4).

**Comparison notebooks (policy — resolves audit 0527-F5).** Companion
*scripts* are the canonical, tested code artifacts (each chapter's
`companions/chXX/{jax,torch}/tests/` pins its numerical claims). *Notebooks*
are an optional secondary layer: curated, cross-framework **comparison
companions** that put the NumPy / JAX / PyTorch idioms side by side as a
teaching narrative. The predecessor `post_transformers` shipped weekly
notebooks; here scripts are primary and notebooks are the curated overlay.
Conventions:

- **Location + naming.** One `.ipynb` per topic under `notebooks/`, named
  `chXX-<topic>.ipynb` (e.g. `notebooks/ch01-matrix-exponential.ipynb`).
- **Output-free.** Source notebooks store **no** cell outputs — the rendered
  HTML is code + markdown only. The chapter prose and its figures are
  canonical; the notebook is a runnable companion, not a figure source.
- **Thin glue over tested code.** Notebooks *import* the companion functions
  (`from companions.chXX.jax import …`) rather than re-implement logic, so
  their substance is covered by the companions' pytest suites. A first cell
  puts the repo root on `sys.path` (mirrors pytest's `pythonpath = .`).
- **Rendering.** `npm run build:notebooks` (book-scaffold `render-notebooks`,
  run on `prebuild`) converts each non-stub `.ipynb` to standalone HTML under
  `public/notebooks/` via `uv run jupyter nbconvert --template basic`; the
  committed HTML is the CI/Cloudflare artifact (the step graceful-skips when
  `uv` is absent). Tooling: `uv pip install -e 'companions/_shared[notebooks]'`.
- **Linking.** Set `notebook_path: notebooks/chXX-<topic>.ipynb` in the chapter
  frontmatter; the chapter header surfaces a "Notebook" link to
  `/notebooks/chXX-<topic>.html`.

---

## 9. Margin notes

**Frequency:** ~3 per chapter, topically tied to the surrounding prose.

**Length:** ~50–80 words typical; **no enforced cap**. This deviates
from `post_transformers/guides/STANDARDS.md`'s 25-word cap, which suits
LaTeX margin rendering; MDX/web flow allows longer notes without
disrupting layout.

**Topics:** technical asides, meta-commentary, connections to the C1
or B pilot, forward/backward chapter references, signposts to the
broader research context.

**What not to put in margin notes:** load-bearing definitions, theorem
statements, exercise prompts. Margin notes are skippable by design.

---

## 10. Voice and pedagogy

**Audience:** specialist (numerical-analysis or dynamical-systems
background, sequence-model researchers). No primers; assume the
reader knows what an SSM is and why they're reading this book.

**Direct-transfer hooks:** name connections to vortex-dynamics,
geophysical fluid dynamics, classical mechanics where they exist.
The book's differentiator is the dynamical-systems lens; surface that
lens whenever it helps.

**Rigor first.** Theorem statements precede their motivation when
the chapter has a load-bearing technical result. Intuition comes
after the formal statement, not before.

**Forward references** are encouraged and should be explicit ("the
exponential-trapezoidal scheme covered in §4.5"); backward references
are mandatory whenever building on prior chapter material.

---

## 11. Authoring checklist for Ch 7–17

Before committing a new chapter, verify:

- [ ] Frontmatter has the 5 required fields (week, part, title,
      status, description).
- [ ] KaTeX macros declared in a frontmatter `{/* */}` comment.
- [ ] §-numbering matches the canonical positional structure:
      content sections → What's next → Exercises → Full solutions →
      Companion code, in that order.
- [ ] Opening `<NoteBox>` titled "Chapter X — at a glance" with Goal +
      Reading time + Direct-transfer hook (or Key insight).
- [ ] All `<Theorem>` blocks have `id="ch##:<type>:<slug>"` or
      `id="ch##:<slug>"` when the slug self-disambiguates.
- [ ] All `<Cite key="...">` keys exist in `bibliography.bib`; new
      bibkeys conform to `<firstauthor><year><firstword>`.
- [ ] All `<Figure>` blocks: `src` under `/figures/ch##/`, alt +
      caption describe the actual figure content (not wished-for
      content), caption credits the producer
      `companions/ch##/<lang>/<script>`.
- [ ] 6 exercises (3 short + 3 long). Short solutions inline via
      `<details>`; long solutions in §X.(second-to-last).
- [ ] Companion code references are narrative; bash/julia invocation
      examples included.
- [ ] ~3 margin notes, topically cohesive, ≤80 words typical.
- [ ] Port-credit headers in any companion file derived from
      post_transformers.
- [ ] `npm run build:bib && npm run build:labels && npm run validate`
      succeeds.
- [ ] No emojis in chapter prose, frontmatter, or commit messages
      (project-wide preference; see audit §3 "Status vocabulary").

---

## 12. Drift and updates

When a new chapter (Ch 7+) introduces a convention this document
doesn't cover, update STYLE.md in the same commit. When a new chapter
deliberately deviates from a convention, add an entry to §13.

The `npm run validate` step in the authoring checklist is the
build-time minimum; deeper lint (bibkey format, theorem-label format)
arrives via audit F6 and F7. STYLE.md states the conventions; the
lints will enforce them.

---

## 13. Known exceptions

- **Ch 1 has 7 exercises** rather than the canonical 6. Ch 1 is
  foundational and carries one extra long-form theory exercise. This
  is the only exercise-count exception; Ch 7+ should target 6 unless
  a similar foundational justification applies.
- **Ch 5 has 5 content sections** (§5.1–§5.5) rather than the canonical
  6, pushing Exercises to §5.7 and Companion code to §5.9. The
  positional rule in §1 accommodates this; the §-numbers in Ch 5 are
  *spec-conforming* via the positional convention, not via a numeric
  override.
- **`mamba_lyapunov_tmlr` bibkey** existed historically as the only
  non-conforming entry in `bibliography.bib`; it was renamed to
  `anonymous2025lyapunov` as part of the F5 landing commit (see
  audit F19's sibling note).
