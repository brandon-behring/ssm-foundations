#!/usr/bin/env node
// check-bibkeys.mjs — validate bibliography.bib + chapter citations.
//
// Two checks:
//   1. Every bibkey in bibliography.bib matches the project convention
//      <firstauthor><year><firstword> — regex ^[a-z]+\d{4}[a-z]+$.
//   2. Every <Cite key="..."> in src/content/chapters/*.mdx resolves to
//      a known bibkey in bibliography.bib.
//
// Documented in STYLE.md §5; closes audit F6.
//
// Exit 0 = clean. Exit 1 = at least one violation; emits all offenders.
//
// Usage:
//   node scripts/check-bibkeys.mjs

import { readFile, readdir } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(__dirname, '..');
const BIB_PATH = path.join(REPO_ROOT, 'bibliography.bib');
const CHAPTERS_DIR = path.join(REPO_ROOT, 'src/content/chapters');

// <firstauthor> = lowercase letters; <year> = 4 digits; <firstword> = letter
// followed by lowercase letters/digits (so `s4`, `mamba2` are accepted).
const BIBKEY_FORMAT = /^[a-z]+\d{4}[a-z][a-z0-9]*$/;

async function loadBibkeys() {
    const bib = await readFile(BIB_PATH, 'utf-8');
    const keys = new Set();
    for (const m of bib.matchAll(/^@\w+\{([^,\s]+)\s*,/gm)) {
        keys.add(m[1]);
    }
    return keys;
}

async function loadCitations() {
    const entries = await readdir(CHAPTERS_DIR);
    const cites = [];
    for (const entry of entries) {
        if (!entry.endsWith('.mdx')) continue;
        const filePath = path.join(CHAPTERS_DIR, entry);
        const mdx = await readFile(filePath, 'utf-8');
        const lines = mdx.split('\n');
        for (let i = 0; i < lines.length; i++) {
            for (const m of lines[i].matchAll(/<Cite\s+key="([^"]+)"/g)) {
                cites.push({ file: entry, line: i + 1, key: m[1] });
            }
        }
    }
    return cites;
}

async function main() {
    const violations = [];

    const bibkeys = await loadBibkeys();
    for (const key of bibkeys) {
        if (!BIBKEY_FORMAT.test(key)) {
            violations.push(
                `bibliography.bib: bibkey \`${key}\` does not match <firstauthor><year><firstword> (regex ${BIBKEY_FORMAT})`,
            );
        }
    }

    const cites = await loadCitations();
    const distinctCiteKeys = new Set(cites.map((c) => c.key));
    for (const cite of cites) {
        if (!bibkeys.has(cite.key)) {
            violations.push(
                `${cite.file}:${cite.line}: <Cite key="${cite.key}" /> not found in bibliography.bib`,
            );
        }
    }

    if (violations.length > 0) {
        console.error('check-bibkeys: violations found');
        for (const v of violations) console.error('  ' + v);
        console.error(`\nTotal: ${violations.length} violation(s).`);
        console.error('See STYLE.md §5 for the bibkey convention.');
        process.exit(1);
    }

    console.log(
        `check-bibkeys: ok (${bibkeys.size} bibkeys, ${cites.length} citations across ${distinctCiteKeys.size} distinct keys)`,
    );
}

main().catch((e) => {
    console.error(e);
    process.exit(1);
});
