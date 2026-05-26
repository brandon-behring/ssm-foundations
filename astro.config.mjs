// @ts-check
/**
 * astro.config.mjs — book-scaffold-astro consumer config (v4 API).
 *
 * Built-in academic style covers Theorem/Lemma/Cite/XRef/Figure + KaTeX + Tufte
 * typography. Override styles by adding to the styles[] array — see
 * recipes/15-defining-styles.md.
 *
 * site: Pass 2 — canonical custom domain. ssm-foundations.brandon-behring.dev
 * was rebound from the legacy bare-name `ssm-foundations` Worker to the
 * person-prefixed `brandon-behring-ssm-foundations` Worker in the Cloudflare
 * dashboard on 2026-05-26 (Phase 1c Pass 2). This URL is now what OG metadata,
 * sitemaps, canonical links, and Pagefind index against. The workers.dev URL
 * (brandon-behring-ssm-foundations.brandon-m-behring.workers.dev) still
 * resolves but should be considered staging, not the public URL.
 *
 * routes.chapters: scaffold v4.3.0+ auto-injects both `/chapters/` index and
 * `/chapters/[...slug]/` dynamic routes. The legacy consumer-owned
 * src/pages/chapters/[...slug].astro was deleted (Commit E) to defer to the
 * scaffold's auto-inject. Per the audit entry, this is Layer 1 of a 3-layer
 * cleanup; DML follows in a future session.
 *
 * title + description (v4.5.0+): feed the auto-injected `/` landing's H1 +
 * lead paragraph + page <meta description>. Mirrored from projects.json on
 * brandon-behring.dev so the landing reads consistently with the portfolio
 * entry. Routed via the v4.5.1 virtual module
 * (virtual:book-scaffold/landing-config) — does NOT collide with .env's
 * BOOK_TITLE line (which remains as harmless tooling-config noise).
 *
 * katexMacros: skipped per upstream-first protocol. The scaffold's default
 * ssmMacros (shipped in v3.6.0) already covers what ssm content needs; add
 * book-specific macros only when content surfaces a missing one.
 */
import { defineBookConfig, academicStyle } from '@brandon_m_behring/book-scaffold-astro';

export default await defineBookConfig({
  site: 'https://ssm-foundations.brandon-behring.dev',
  title: 'ssm-foundations',
  description: '17-chapter lens-led foundations book bridging numerical analysis and dynamical systems to modern sequence-model architectures (SSMs, Mamba, S4, Hyena, DeltaNet).',
  styles: [academicStyle],
  output: 'static',
  routes: { chapters: true },
});
