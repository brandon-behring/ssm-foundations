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
