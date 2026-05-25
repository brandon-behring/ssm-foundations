# Companions

Per-chapter code companions. Layout:

```
companions/
  chXX/
    jax/      # JAX/Flax — primary language
    julia/    # Julia (DifferentialEquations.jl, ControlSystems.jl)
    torch/    # PyTorch — for cross-language reference where useful
```

Inline-companion routing is wired via book-scaffold-astro's inline-companions feature (v3.2+). Add `companions:` to chapter frontmatter to surface code blocks in the chapter sidebar.

**Current state (2026-05-25)**: Ch 1–6 populated — JAX companions across all six chapters, Julia companions in Ch 4–6 (with per-chapter `runtests.jl` and a shared `_shared/JuliaFormatter.toml`). Ch 7–17 directories created on-demand as chapters author.

## Port-credit convention

When a companion file is derived (in whole or substantively) from code in `post_transformers/`, its first docstring or comment block must cite the source with an absolute GitHub URL pinned to `main`, per the [cross-repo link convention](../CLAUDE.md). Lived examples:

- [`_shared/plot_utils.py`](_shared/plot_utils.py) — opens with "Minimal port of ``post_transformers/guides/shared/plot_utils.py``".
- [`ch04/julia/discretization_atlas.jl`](ch04/julia/discretization_atlas.jl) — opens with "Ported and extended from ``post_transformers/experiments/julia/week09/discretization.jl``".

The convention exists to preserve provenance as the C1 + B pilots port more code from the research sibling. A future lint (`scripts/check-port-credits.mjs`, audit F6/F7 sibling) may enforce it; for now it is honor-system. Ports should also document what was *changed* from the source (precision, system parameters, additional exports) so the relationship to upstream stays legible.
