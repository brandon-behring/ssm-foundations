#!/usr/bin/env node
// check-xref-labels.mjs — validate <Theorem id="..."> and <Figure id="..."> format.
//
// Per STYLE.md §4, IDs on <Theorem> and <Figure> follow either:
//   - ch##:<type>:<slug>  where <type> is a short type abbreviation
//                          (def | thm | prop | lemma | ex | rem)
//                          and <slug> is short-kebab-case
//   - ch##:<slug>         when the slug self-disambiguates
//
// Closes audit F7. Exit 0 = clean; exit 1 = at least one violation.
//
// Usage:
//   node scripts/check-xref-labels.mjs

import { readFile, readdir } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(__dirname, '..');
const CHAPTERS_DIR = path.join(REPO_ROOT, 'src/content/chapters');

// Accepts: ch##:<slug>  or  ch##:<type>:<slug>
// where <slug> is short-kebab-case (letters/digits/hyphen).
// <type> when present is one of the abbreviated theorem types.
const ID_FORMAT = /^ch\d{2}:([a-z]+:)?[a-z0-9][a-z0-9-]*$/;
const ALLOWED_TYPE_PREFIXES = new Set(['def', 'thm', 'prop', 'lemma', 'ex', 'rem', 'fig']);

async function main() {
    const violations = [];
    const seenIds = new Map(); // id -> [{file, line}]

    const entries = await readdir(CHAPTERS_DIR);
    for (const entry of entries) {
        if (!entry.endsWith('.mdx')) continue;
        const filePath = path.join(CHAPTERS_DIR, entry);
        const mdx = await readFile(filePath, 'utf-8');
        const lines = mdx.split('\n');

        for (let i = 0; i < lines.length; i++) {
            const line = lines[i];
            const isTheoremOrFigure =
                line.includes('<Theorem') || line.includes('<Figure');
            if (!isTheoremOrFigure) continue;

            const idMatch = line.match(/\bid="([^"]+)"/);
            if (!idMatch) continue; // No id; not required (yet)
            const id = idMatch[1];

            if (!ID_FORMAT.test(id)) {
                violations.push(
                    `${entry}:${i + 1}: id="${id}" does not match ch##:[<type>:]<slug> (regex ${ID_FORMAT})`,
                );
                continue;
            }

            // If a type prefix is present, check it's one of the allowed values.
            const parts = id.split(':');
            if (parts.length === 3) {
                const type = parts[1];
                if (!ALLOWED_TYPE_PREFIXES.has(type)) {
                    violations.push(
                        `${entry}:${i + 1}: id="${id}" uses unknown type prefix "${type}"; allowed: ${[...ALLOWED_TYPE_PREFIXES].sort().join(', ')}`,
                    );
                }
            }

            // Track duplicates across the whole repo.
            if (!seenIds.has(id)) seenIds.set(id, []);
            seenIds.get(id).push({ file: entry, line: i + 1 });
        }
    }

    for (const [id, occ] of seenIds) {
        if (occ.length > 1) {
            const locs = occ.map((o) => `${o.file}:${o.line}`).join(', ');
            violations.push(`duplicate id="${id}" at ${locs}`);
        }
    }

    if (violations.length > 0) {
        console.error('check-xref-labels: violations found');
        for (const v of violations) console.error('  ' + v);
        console.error(`\nTotal: ${violations.length} violation(s).`);
        console.error('See STYLE.md §4 for the cross-reference ID convention.');
        process.exit(1);
    }

    const total = [...seenIds.values()].reduce((s, occ) => s + occ.length, 0);
    console.log(
        `check-xref-labels: ok (${seenIds.size} distinct IDs across ${total} occurrences)`,
    );
}

main().catch((e) => {
    console.error(e);
    process.exit(1);
});
