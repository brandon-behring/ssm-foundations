// @ts-check
/**
 * astro.config.mjs — book-scaffold-astro consumer config (v4 API).
 *
 * Built-in academic style covers Theorem/Lemma/Cite/XRef/Figure + KaTeX + Tufte
 * typography. Override styles by adding to the styles[] array — see
 * recipes/15-defining-styles.md.
 */
import { defineBookConfig, academicStyle } from '@brandon_m_behring/book-scaffold-astro';

export default await defineBookConfig({
  styles: [academicStyle],
  site: 'https://ssm-foundations.brandonbehring.workers.dev',
});
