# Chapter research brief — Ch NN: «short title»

> **What this is.** A lightweight per-chapter context profile, filled in *before*
> authoring starts (the `/exploring-options` input). Generalizes the Ch 11
> pre-recon. Copy to `docs/briefs/chNN-<slug>.md` (or keep in project-memory),
> fill the bracketed prompts, then drive `/exploring-options` from §6.
> Delete a section only if you've confirmed it's genuinely empty — an empty
> "Forward-promises" is a claim, not an omission.

- **Chapter / slug:** ch NN — `chNN-<slug>.mdx`
- **Part / status target:** «foundations | ssm-core | beyond-ssm | integration | synthesis» → `implemented`
- **One-line scope:** «what this chapter teaches, and the one idea a reader leaves with»
- **Pilot tie-in:** «C1 (symplectic) / B (two-timescale) / none — and what the pilot needs from this chapter»

## 1. Forward-promises to redeem
Earlier chapters that *promised* this chapter would cover something. Each must be
honoured or explicitly renegotiated. (`grep -rn "Chapter NN" src/content/chapters/`.)

| Source (file:line) | Promise made | How this chapter honours it |
|---|---|---|
| `chNN:LL` | «quoted promise» | «section / renegotiation» |

## 2. Backward-reference anchors (→ `<XRef>` targets)
Labelled objects in *prior* chapters this chapter cites. Backward refs to existing
ids use `<XRef id="…" />` (STYLE.md §4); confirm each id resolves in `labels.json`.

| Target id | What it is | Where this chapter leans on it |
|---|---|---|
| `chMM:thm:slug` | «theorem name» | «§ that invokes it» |

## 3. Predecessor reuse
What exists in `post_transformers` (and elsewhere) to port vs. write fresh.

- **High reuse (port):** «`post_transformers/experiments/jax/weekNN/<file>.py` — what it gives»
- **Greenfield (author from paper math):** «components with no usable predecessor»
- **Reference-only (anchor, don't copy):** «dossiers / external repos»

## 4. Bibliography adds
Run `grep -c '^@' bibliography.bib` before/after; `npm run build:bib` after edits.

- **Present:** «bibkeys already in `bibliography.bib`»
- **To add (`<firstauthor><year><firstword>`):** «key — Author Year, arXiv:XXXX»
- **Defer to a later chapter:** «keys that belong downstream»
- **gitleaks watch:** high-entropy bibkeys (entropy > 3.5) trip the commit hook → append the
  printed `chNN-slug.mdx:generic-api-key:LINE` to `.gitleaksignore` and re-commit.

## 5. Scope tensions / boundaries
The "what's in vs. out" calls — especially any a *prior* chapter's wording pre-committed.

- «Tension: earlier text implied X belongs here, but X is really Ch MM. Resolution: hand forward.»

## 6. Decisions for `/exploring-options`
The open design choices to settle interactively before drafting.

1. **Companion languages.** JAX is canonical (always). **torch parity is required for
   architecture chapters** (the 0527-F27 policy); Julia only where SciML adds pedagogy. Figures: JAX-produced.
2. **Section-weight allocation:** «how to split the N content sections»
3. **Entry framing:** «derive from scratch vs. lean on a prior theorem»
4. **«chapter-specific decision»**

## 7. Likely chapter shape (sketch)
Positional skeleton (STYLE.md §1): content §NN.1…§NN.k → What's next → Exercises → Full solutions →
Companion code.

1. §NN.1 «…»
2. …
3. §NN.k «…»

## 8. Companion plan
- **JAX** (`companions/chNN/jax/`): «scripts + figures each produces»
- **torch** (`companions/chNN/torch/`): «parity ports — compute+parity only, no figures»
- **Julia** (`companions/chNN/julia/`): «only if it earns its place»
- **Tests** (STYLE.md §8 bar): exact identities `< 1e-12`, JAX↔torch parity `< 1e-9` float64, each figure's
  load-bearing quantity pinned; `--import-mode=importlib` (no `__init__.py`).

## 9. Gate items + gotchas
- Run the chapter through `claim-skeptic` + `chapter-auditor` before advancing `status:`.
- Companions-first (prose cites *measured* numbers from companion stdout — F19 caption-truthfulness).
- Commit-time gitleaks (§4); explicit `git push`; new `feat/chNN-slug` branch; torch tests run **without**
  `PYTHONPATH=.` (use the Makefile targets); re-derive any predecessor port that looks subtly wrong.
- After shipping: update `CURRENT_WORK.md`, the roadmap memory, and `docs/DASHBOARD.md`.
