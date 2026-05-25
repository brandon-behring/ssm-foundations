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

**Current state (2026-05-24)**: empty per-language subdirs for Ch 1–3 (Phase 3 will populate). Ch 4–17 directories will be created on-demand as chapters are authored.
