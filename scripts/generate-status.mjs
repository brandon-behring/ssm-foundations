#!/usr/bin/env node
// generate-status.mjs — auto-regenerate docs/STATUS.md from chapter state.
//
// Walks src/content/chapters/*.mdx, parses frontmatter, and counts content
// markers (exercises, citations, figures, companion files). Emits a
// canonical per-chapter table to docs/STATUS.md with a `verified` date
// header that the future make status-check target (audit F10) will
// staleness-gate at <=14 days.
//
// Closes audit F11 — the durable fix for the truthfulness debt F1
// only patched.
//
// Usage:
//   node scripts/generate-status.mjs        # writes docs/STATUS.md
//   node scripts/generate-status.mjs --check  # exits 1 if STATUS.md is stale

import { readFile, readdir, writeFile, mkdir } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(__dirname, '..');
const CHAPTERS_DIR = path.join(REPO_ROOT, 'src/content/chapters');
const COMPANIONS_DIR = path.join(REPO_ROOT, 'companions');
const STATUS_PATH = path.join(REPO_ROOT, 'docs/STATUS.md');
const STALENESS_DAYS = 14;

const STATUS_ORDER = [
    'implemented',
    'chapter_only',
    'reading_only',
    'prose_only',
    'code_only',
    'scaffolded',
    'planned',
];

function parseFrontmatter(mdx) {
    const match = mdx.match(/^---\n([\s\S]*?)\n---/);
    if (!match) return {};
    const fm = {};
    for (const line of match[1].split('\n')) {
        const m = line.match(/^(\w+):\s*(.*)$/);
        if (m) {
            let v = m[2].trim();
            if (v.startsWith("'") && v.endsWith("'")) v = v.slice(1, -1);
            if (v.startsWith('"') && v.endsWith('"')) v = v.slice(1, -1);
            fm[m[1]] = v;
        }
    }
    return fm;
}

function countMarkers(mdx) {
    const lines = mdx.split('\n').length;
    const exerciseHeadings = (mdx.match(/^### Exercise \d/gm) || []).length;
    const solutionHeadings = (mdx.match(/^### Solution to Exercise/gm) || []).length;
    const figures = (mdx.match(/<Figure\s/g) || []).length;
    const theorems = (mdx.match(/<Theorem\s/g) || []).length;
    const marginNotes = (mdx.match(/<MarginNote>/g) || []).length;
    const citeKeys = new Set();
    for (const m of mdx.matchAll(/<Cite\s+key="([^"]+)"/g)) {
        citeKeys.add(m[1]);
    }
    return {
        lines,
        exerciseHeadings,
        solutionHeadings,
        figures,
        theorems,
        marginNotes,
        distinctCitations: citeKeys.size,
    };
}

async function countCompanionFiles(chSlug) {
    // chSlug like 'ch01' (just the prefix)
    const result = { jax: 0, julia: 0, torch: 0 };
    for (const lang of ['jax', 'julia', 'torch']) {
        const dir = path.join(COMPANIONS_DIR, chSlug, lang);
        if (!existsSync(dir)) continue;
        try {
            const entries = await readdir(dir);
            for (const e of entries) {
                if (e === '__pycache__' || e.startsWith('.')) continue;
                if (e.endsWith('.py') || e.endsWith('.jl')) {
                    result[lang]++;
                }
            }
        } catch {
            // ignore
        }
    }
    return result;
}

async function collectChapters() {
    const entries = await readdir(CHAPTERS_DIR);
    const chapters = [];
    for (const entry of entries) {
        if (!entry.endsWith('.mdx')) continue;
        const m = entry.match(/^(ch\d+)-(.+)\.mdx$/);
        if (!m) continue;
        const filePath = path.join(CHAPTERS_DIR, entry);
        const mdx = await readFile(filePath, 'utf-8');
        const fm = parseFrontmatter(mdx);
        const markers = countMarkers(mdx);
        const companions = await countCompanionFiles(m[1]);
        chapters.push({
            slug: m[1],
            week: parseInt(fm.week || '0', 10),
            title: fm.title || '',
            status: fm.status || 'unknown',
            ...markers,
            companions,
        });
    }
    chapters.sort((a, b) => a.week - b.week);
    return chapters;
}

function statusRollup(chapters) {
    const counts = {};
    for (const s of STATUS_ORDER) counts[s] = 0;
    for (const c of chapters) {
        counts[c.status] = (counts[c.status] || 0) + 1;
    }
    return counts;
}

function renderStatusMd(chapters, verifiedDate) {
    const rollup = statusRollup(chapters);
    const totalImplemented = rollup.implemented || 0;
    const totalChapters = chapters.length;
    const totalExercises = chapters.reduce((s, c) => s + c.exerciseHeadings, 0);
    const totalCitations = new Set();
    for (const c of chapters) {
        // distinct citations across all chapters: re-read for the union
    }
    // For overall distinct citations, we'd need to re-walk — keep per-chapter
    // distinct in the table and total = sum of per-chapter sizes (overestimate
    // when chapters share keys; STATUS.md notes this).
    const sumCitations = chapters.reduce((s, c) => s + c.distinctCitations, 0);
    const sumFigures = chapters.reduce((s, c) => s + c.figures, 0);
    const sumTheorems = chapters.reduce((s, c) => s + c.theorems, 0);
    const sumCompanions = chapters.reduce(
        (s, c) => s + c.companions.jax + c.companions.julia + c.companions.torch,
        0,
    );

    let md = '# Status snapshot — ssm-foundations\n\n';
    md += `**Verified:** ${verifiedDate}\n\n`;
    md += 'Auto-generated by `scripts/generate-status.mjs`. Do not edit by hand; ';
    md += 're-run via `node scripts/generate-status.mjs` (or `make status-snapshot` ';
    md += 'once audit F10 lands). The future `make status-check` target will ';
    md += `fail if this file's verified date is more than ${STALENESS_DAYS} days stale.\n\n`;

    md += '## Per-chapter\n\n';
    md += '| Ch | Status | Lines | Theorems | Exercises | Solutions | Figures | Cites | JAX | Julia | torch |\n';
    md += '|----|--------|-------|----------|-----------|-----------|---------|-------|-----|-------|-------|\n';
    for (const c of chapters) {
        md += `| ${c.slug} | \`${c.status}\` | ${c.lines} | ${c.theorems} | ${c.exerciseHeadings} | ${c.solutionHeadings} | ${c.figures} | ${c.distinctCitations} | ${c.companions.jax} | ${c.companions.julia} | ${c.companions.torch} |\n`;
    }

    md += '\n## Rollup\n\n';
    md += `- **Chapters:** ${totalChapters} total; ${totalImplemented} \`implemented\``;
    const planned = rollup.planned || 0;
    if (planned > 0) md += `, ${planned} \`planned\``;
    md += '.\n';
    md += '- **Status breakdown:**';
    let first = true;
    for (const s of STATUS_ORDER) {
        if (rollup[s] > 0) {
            md += first ? ' ' : ', ';
            md += `\`${s}\`: ${rollup[s]}`;
            first = false;
        }
    }
    md += '.\n';
    md += `- **Content totals (sum across chapters):** ${sumTheorems} theorems, ${sumExercisesValue(chapters)} exercises (with ${chapters.reduce((s, c) => s + c.solutionHeadings, 0)} full solutions), ${sumFigures} figures, ${sumCitations} citation slots (per-chapter distinct; overall distinct may be smaller).\n`;
    md += `- **Companion files (.py + .jl, excluding __pycache__):** ${sumCompanions} total across all chapters.\n\n`;

    md += '## Notes\n\n';
    md += '- The `Cites` column is the number of *distinct* `<Cite key="...">` keys ';
    md += 'within each chapter. The overall total is a sum (overestimate when chapters ';
    md += 'share citations); the canonical bibliography lives in `bibliography.bib` ';
    md += '(16 entries as of this snapshot).\n';
    md += '- Companion counts ignore `__pycache__/` and dotfiles. Julia companions ';
    md += 'live in `companions/chXX/julia/` and include `runtests.jl` where present ';
    md += '(see audit F8 / `companions/_shared/JuliaFormatter.toml`).\n';
    md += '- Status taxonomy is the 7-state from CLAUDE.md §"Status taxonomy (7-state)".\n';

    return md;
}

function sumExercisesValue(chapters) {
    return chapters.reduce((s, c) => s + c.exerciseHeadings, 0);
}

// Return the first line where two markdown strings diverge, or null if equal.
// Used by both --check and the script test to point at the exact drift.
function firstDivergence(expected, actual) {
    const exp = expected.split('\n');
    const got = actual.split('\n');
    const n = Math.max(exp.length, got.length);
    for (let i = 0; i < n; i++) {
        if (exp[i] !== got[i]) {
            return { line: i + 1, expected: exp[i] ?? null, got: got[i] ?? null };
        }
    }
    return null;
}

async function check() {
    if (!existsSync(STATUS_PATH)) {
        console.error('docs/STATUS.md does not exist; run `node scripts/generate-status.mjs` to create it.');
        process.exit(1);
    }
    const content = await readFile(STATUS_PATH, 'utf-8');
    const m = content.match(/\*\*Verified:\*\*\s+(\d{4}-\d{2}-\d{2})/);
    if (!m) {
        console.error('docs/STATUS.md missing a `**Verified:** YYYY-MM-DD` line.');
        process.exit(1);
    }
    const verifiedDate = m[1];

    // Content check (audit #26): regenerate the snapshot from current chapter
    // state, reusing the committed verified date so the `**Verified:**` line
    // also matches, then require a byte-identical result. Any divergence is a
    // genuine content drift — a stale `status` column, a missing/orphan row, a
    // changed count, or a rollup mismatch. Because the same regex runs on both
    // sides, count edge-cases (e.g. a `<Theorem` inside a `{/* */}` comment)
    // can't produce a false diff: they're counted identically either way.
    const chapters = await collectChapters();
    const expected = renderStatusMd(chapters, verifiedDate);
    const div = firstDivergence(expected, content);
    if (div) {
        console.error('docs/STATUS.md content is stale — it no longer matches src/content/chapters/.');
        console.error(`  first divergence at line ${div.line}:`);
        console.error(`    committed: ${JSON.stringify(div.got)}`);
        console.error(`    expected:  ${JSON.stringify(div.expected)}`);
        console.error('Re-run `node scripts/generate-status.mjs` to regenerate docs/STATUS.md.');
        process.exit(1);
    }

    // Staleness check (original behaviour): even a content-fresh snapshot is
    // flagged if its verified date is too old. Local midnight — must match the
    // local-date stamp written by main() (PR #21 review fix).
    const verified = new Date(verifiedDate + 'T00:00:00');
    const ageDays = (Date.now() - verified.getTime()) / (1000 * 60 * 60 * 24);
    if (ageDays > STALENESS_DAYS) {
        console.error(
            `docs/STATUS.md is ${ageDays.toFixed(1)} days stale (verified ${verifiedDate}, limit ${STALENESS_DAYS}).`,
        );
        console.error('Re-run `node scripts/generate-status.mjs` to refresh.');
        process.exit(1);
    }
    console.log(
        `status-check: ok (content matches; verified ${verifiedDate}, age ${ageDays.toFixed(1)}d ≤ ${STALENESS_DAYS}d)`,
    );
}

async function main() {
    const args = process.argv.slice(2);
    if (args.includes('--check')) {
        await check();
        return;
    }
    const chapters = await collectChapters();
    // Local-timezone date (en-CA locale = YYYY-MM-DD): a late-evening regen must
    // not stamp tomorrow's UTC date and desync from hand-dated docs (PR #21 review).
    const today = new Date().toLocaleDateString('en-CA');
    const md = renderStatusMd(chapters, today);
    await mkdir(path.dirname(STATUS_PATH), { recursive: true });
    await writeFile(STATUS_PATH, md);
    console.log(`generate-status: wrote docs/STATUS.md (${chapters.length} chapters, verified ${today})`);
}

// Only auto-run when invoked directly (node scripts/generate-status.mjs);
// stay side-effect-free when imported by scripts/generate-status.test.mjs.
const isEntryPoint = import.meta.url === pathToFileURL(process.argv[1] ?? '').href;
if (isEntryPoint) {
    main().catch((e) => {
        console.error(e);
        process.exit(1);
    });
}

export { collectChapters, renderStatusMd, firstDivergence, STATUS_PATH, STALENESS_DAYS };
