# ssm-foundations

A lens-led 17-chapter foundations book for post-transformer sequence-model architectures, foregrounding the dynamical-systems perspective: continuous-time math first, then discretization theory, then the SSM family and its delta-rule / gating / hybrid cousins.

> **Status — alpha (2026-05-24)**: scaffold landed. Chapters 1–6 are authored (`status: implemented`); Ch 7–17 are planned stubs. Expect breaking changes to structure and prose. Pre-release banner is live site-wide. Substantive feedback welcome via issues.

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
