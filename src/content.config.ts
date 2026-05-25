/**
 * src/content.config.ts — Content collections.
 * defineBookSchemas returns chapters + tools-collateral; extend via
 * standard JS spread + Zod `.extend()` if you need book-specific fields.
 */
import { defineBookSchemas } from '@brandon_m_behring/book-scaffold-astro/schemas';

export const { collections } = defineBookSchemas();
