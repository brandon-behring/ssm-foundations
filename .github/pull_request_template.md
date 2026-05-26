<!--
Thank you for the PR! Please confirm the relevant boxes below. The
STYLE.md authoring checklist (§11) is the canonical pre-commit gate.
-->

## Summary

<!-- Two or three sentences on what changed and why. Link to the issue
this resolves if applicable. -->

## Audit reference

<!-- If this PR addresses an audit finding, name it (e.g., "Closes audit F19"
or "Partial fix for F11"). Otherwise delete this section. -->

## Authoring checklist (chapter / prose PRs)

- [ ] Frontmatter has the 5 required fields (week, part, title, status, description)
- [ ] KaTeX macros declared in a frontmatter `{/* */}` comment
- [ ] Section numbering follows the positional rule (Exercises = 3rd-to-last, Solutions = 2nd-to-last, Companion = last; canonical N=6)
- [ ] All `<Theorem>` blocks have `id="ch##:[<type>:]<slug>"`
- [ ] All `<Cite key="...">` keys exist in `bibliography.bib`
- [ ] All `<Figure>` captions describe the *actual* figure content (per audit F19)
- [ ] 6 exercises (3 short + 3 long), or documented exception in STYLE.md §13
- [ ] Companion code references are narrative; bash/julia invocation examples included
- [ ] Port-credit headers in any companion file derived from post_transformers
- [ ] No emojis (project-wide preference; STYLE.md §11 checklist)

## Companion-code checklist (only for `companions/` PRs)

- [ ] Real `@test` assertions in `runtests.jl`, not smoke wraps (audit F8)
- [ ] Port-credit header citing source URL pinned to `main` (audit F4)
- [ ] Magnitudes in captions / docstrings match empirical output (audit F19)

## Local verification

```bash
make check     # validate + lint + status-check
```

Confirm `make check` exits 0 before merging.
