# ssm-foundations

A lens-led 17-chapter foundations book for post-transformer sequence-model architectures, foregrounding the dynamical-systems perspective: continuous-time math first, then discretization theory, then the SSM family and its delta-rule / gating / hybrid cousins.

> **Status — alpha (updated 2026-06-05)**: Chapters 1–11 are authored (`status: implemented`) — the foundations (1–6) and SSM-core (7–10) lines plus the first beyond-SSM chapter (11, linear attention + Hyena); Ch 12–17 are planned stubs. Expect breaking changes to structure and prose. Pre-release banner is live site-wide. Substantive feedback welcome via issues.

---

## Who this is for

- Working sequence-model researchers who want the math foundations explicit, not assumed.
- ML practitioners with a numerical-analysis or dynamical-systems background who want the SSM literature framed in their native vocabulary.
- Anyone choosing between attention, SSM, delta-rule, gated-linear, and hybrid architectures and wanting a non-empirical-only basis for the choice.

---

## Reading paths

- **Foundations only** (Ch 1–6): a self-contained primer on continuous-time linear systems, stability theory, structured linear algebra, discretization theory, stability regions, and implicit/symplectic integration. Useful even without the SSM chapters.
- **Full curriculum** (Ch 1–17): the lens-led 17-chapter design — see [foundations §3](https://github.com/brandon-behring/post_transformers/blob/main/notes/foundations_curriculum_design_2026_05_20.md) for chapter-at-a-glance.
- **Niche-focused**: see Ch 17 (niche-pilot integration) for which chapter subsets the active C1 (symplectic integrators) and B (two-timescale benchmarks) research pilots draw from.

---

## Chapters

17 chapters in five parts. `Status` is the 7-state taxonomy (defined in `CLAUDE.md`); the live per-chapter metric snapshot (lines, theorems, exercises, companions) lives in [`docs/STATUS.md`](docs/STATUS.md).

| Ch | Status | Part | Topic |
|----|--------|------|-------|
| 1 | `implemented` | foundations | Linear ODEs as state-space systems |
| 2 | `implemented` | foundations | Stability theory: Lyapunov, A-stability, BIBO |
| 3 | `implemented` | foundations | Structured linear algebra and conditioning |
| 4 | `implemented` | foundations | Discretization: ZOH, bilinear, exponential families |
| 5 | `implemented` | foundations | Stability regions, Butcher tableau, order conditions |
| 6 | `implemented` | foundations | Implicit methods, stiff systems, symplectic integration |
| 7 | `implemented` | ssm-core | HiPPO: orthogonal-basis projection operators |
| 8 | `implemented` | ssm-core | LTI SSMs: S4, S4D, S5 |
| 9 | `implemented` | ssm-core | Selective SSMs: Mamba-1, Mamba-2, SSD |
| 10 | `implemented` | ssm-core | Mamba-3 and the exponential-trapezoidal integrator |
| 11 | `implemented` | beyond-ssm | Linear attention and Hyena |
| 12 | `planned` | beyond-ssm | Delta-rule lineage: DeltaNet, Gated DeltaNet, Kimi Linear |
| 13 | `planned` | beyond-ssm | Exponential gates and matrix memory: xLSTM, RWKV-7 |
| 14 | `planned` | integration | Hybrid architectures and gating mechanisms |
| 15 | `planned` | integration | Counter-evidence and diagnostic tools |
| 16 | `planned` | integration | Empirical methodology: benchmarks and evaluation |
| 17 | `planned` | synthesis | Niche-pilot integration |

---

## Lineage

- **Predecessor**: [`post_transformers`](https://github.com/brandon-behring/post_transformers) — 21-week curriculum (`roadmap.md`), dossier folders, and research-pilot kickoffs. That repo stays canonical for week-numbered cross-references inside the research ecosystem; this book is the lens-led public reorganization.
- **Design doc**: [`foundations_curriculum_design_2026_05_20.md`](https://github.com/brandon-behring/post_transformers/blob/main/notes/foundations_curriculum_design_2026_05_20.md) — the 17-chapter design, bidirectional crosswalk to W1–W21, open-questions ledger.
- **Niche commitment**: [`niche_decision_2026_05_24.md`](https://github.com/brandon-behring/post_transformers/blob/main/notes/niche_decision_2026_05_24.md) — the C1 + B research direction this book backstops.

---

## Author

Brandon Behring (PhD applied math, NJIT 2020; vortex-dynamics specialization). The book's dynamical-systems framing draws on the user's research background applying symplectic integrators to Hamiltonian systems.

---

## Build

```bash
npm install
npm run dev               # localhost:4321
npm run build             # → dist/
npx wrangler deploy       # → Cloudflare Workers
```

Built with [`@brandon_m_behring/book-scaffold-astro`](https://github.com/brandon-behring/book-scaffold-astro) (academic preset, v4.8.0).
