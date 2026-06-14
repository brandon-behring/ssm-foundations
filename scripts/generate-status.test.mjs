// generate-status.test.mjs — real tests for the audit-#26 content check.
//
// Run: node --test scripts/generate-status.test.mjs  (or `make test-scripts`).
//
// These exercise the same invariant `generate-status.mjs --check` enforces in
// CI — that docs/STATUS.md is a faithful render of src/content/chapters/ — so
// they double as a regression guard on the snapshot itself.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

import {
    collectChapters,
    renderStatusMd,
    firstDivergence,
    STATUS_PATH,
} from './generate-status.mjs';

const committed = await readFile(STATUS_PATH, 'utf-8');
const verified = committed.match(/\*\*Verified:\*\*\s+(\d{4}-\d{2}-\d{2})/)?.[1];
const chapters = await collectChapters();

test('committed STATUS.md matches a fresh render (content is in sync)', () => {
    assert.ok(verified, 'docs/STATUS.md has a `**Verified:** YYYY-MM-DD` line');
    const expected = renderStatusMd(chapters, verified);
    const div = firstDivergence(expected, committed);
    assert.equal(
        div,
        null,
        div &&
            `STATUS.md drifted at line ${div.line}:\n  committed: ${JSON.stringify(div.got)}\n  expected:  ${JSON.stringify(div.expected)}\n  → run \`node scripts/generate-status.mjs\``,
    );
});

test('a stale status column is detected', () => {
    // Simulate frontmatter advancing without a STATUS.md regen: flip one
    // chapter's status in the freshly-collected set and confirm the render no
    // longer matches the committed file.
    const mutated = chapters.map((c, i) => (i === 0 ? { ...c, status: 'scaffolded' } : c));
    const div = firstDivergence(renderStatusMd(mutated, verified), committed);
    assert.notEqual(div, null, 'a changed status must diverge from the committed snapshot');
});

test('a missing chapter row is detected', () => {
    // Drop a chapter (row-set drift) and confirm the render diverges.
    const fewer = chapters.slice(1);
    const div = firstDivergence(renderStatusMd(fewer, verified), committed);
    assert.notEqual(div, null, 'a missing row must diverge from the committed snapshot');
});

test('firstDivergence returns null for identical strings', () => {
    assert.equal(firstDivergence('a\nb\nc', 'a\nb\nc'), null);
    assert.deepEqual(firstDivergence('a\nb', 'a\nB'), { line: 2, expected: 'b', got: 'B' });
});
