# ssm-foundations — Audit Reconciliation Pass

**Audit date:** 2026-05-29
**Type:** Reconciliation (read-only; reconciles prior audits against `HEAD`)
**HEAD at audit:** `966b619` ("feat: adopt scaffold v4.8.0 + backfill ch01 provenance", 2026-05-28)
**Scope:** all prior findings across the three audits below, verified against current repo state. No fixes applied — this pass classifies, it does not remediate.
**Reconciles:**
- `audits/2026-05-25_standards_vs_post_transformers.md` (inaugural; F1–F19)
- `audits/2026-05-26_first-deploy.md` (deploy decisions + open items)
- `audits/2026-05-27_repo_audit_deeper.md` (deep pass; F0–F37)
- `audits/2026-05-27_repo_content_quality_audit.md` (draft; A1–A10 — `[SUPERSEDED]`)
**Format:** F-numbered, Track A/B/C, bracketed text status (per `project_ssm_foundations_audit_format` memory). No emoji (per `feedback_no_emojis_in_artifact_tracking` memory).

---

## Remediation status (live — updated 2026-05-29)

Track A remediation executed on branch `remediation/track-a`. Status of findings against that branch:

- **[RESOLVED] (Track A):** 0527-F1, F2, F3, F6, F8, F11, F12, F18, F19, F20, F30, F31, F32, F35, F36 (help-text portion); 0529-F1, F4. Each `remediation/track-a` commit cites `Closes 0527-Fn`.
- **[DEFERRED → Phase 7] (needs runnable JAX/matplotlib env):** 0527-F34 (Ex 6.3 Δ=0.05 drift reproducibility), 0527-F14 (HiPPO-κ wording + figure-vs-claim check).
- **[OPEN → Track B]:** 0527-F4 (XRef enforcement), F7 + F36 (Julia ch04 gate, full close), F26 (JAX pytest), F27 (torch); Manifest policy; stale remote branch. Tracked via the umbrella issue.
- **[DISPUTED — no change]:** 0527-F16.
- **[DEFERRED-C — unchanged]:** the Track-C rows in §1.3.

The per-finding rows in §1.3 below are the original 2026-05-29 snapshot; this section is the authoritative live status.

---

## 0. Why this pass exists

The three prior audits constitute a sophisticated finding-tracking system (~106 KB, F-numbered, Track A/B/C, bracketed status). But the system **was never reconciled against the commits that followed it**, so its status markers are now wrong in *both* directions, and a naive reading materially misleads:

1. **Falsely open.** `0527-F29` / `0527-F33` (Julia exp-trap math bug + missing test) carry `[open]` in the 2026-05-27 audit, but commit `08f7e7c` fixed both the next day. The inaugural audit's entire infra slate (`0525-F5`–`F11`) and even all of its Track-C deferred debt (`0525-F12`–`F16`: LICENSE, CONTRIBUTING.md, CI, issue/PR templates, .editorconfig) are done — the files exist now.
2. **Falsely closed / invisible.** Five `implemented` chapters (ch02–06) never received the v4.8.0 provenance block; CLAUDE.md/README doc-drift *worsened* to six scaffold versions behind; and the two F-numbered audits share a **colliding namespace** (`0525-F8` = "Julia+torch rigor"; `0527-F8` = "version drift").
3. **Line-ref rot.** Commit `966b619` inserted a 10-line provenance block into ch01, so every audit line-reference into that file is now off by ~10 (e.g. `0527-F12` cites ch01:111; the claim now lives at ch01:121).

This pass establishes the single reconciled source of truth and declares the **forward F-number registry convention** (§6, `0529-F2`).

### Status vocabulary (this pass)

- `[RESOLVED]` — fixed; commit SHA or now-present file cited.
- `[OPEN]` — verified still present against `HEAD`.
- `[OPEN-STALE-REF]` — finding holds, but its audit line-refs drifted (re-anchored here).
- `[PARTIAL]` — partly resolved.
- `[DISPUTED]` — prior audit may have over-flagged; reasoning given.
- `[DEFERRED-C]` — Track C, intentionally not-yet-done.
- `[EXTERNAL]` — depends on another repo or the Cloudflare dashboard; not actionable in-repo.
- `[SUPERSEDED]` / `[WITHDRAWN]` — historical.

---

## 1. Reconciliation table

### 1.1 Inaugural audit (`0525-F1` … `0525-F19`)

| ID | Finding (short) | Status | Evidence (against HEAD) |
|---|---|---|---|
| 0525-F1 | CLAUDE.md/README/companions status drift | `[PARTIAL]` | `companions/README.md:15` now accurate; **but CLAUDE.md:5 prose-context + README.md:5 still stale** — folds into `0527-F1`/`0529-F4` |
| 0525-F2 | CLAUDE.md missing hub-reference block | `[RESOLVED]` | `CLAUDE.md` "Hub Pattern References" block present (lines ~95–101) |
| 0525-F3 | No `audits/` directory | `[RESOLVED]` | `audits/` exists (5 files incl. this one) |
| 0525-F4 | Port-credit convention undocumented | `[RESOLVED]` | `companions/README.md` port-credit section present |
| 0525-F5 | No consolidated `STYLE.md` | `[RESOLVED]` | `STYLE.md` present |
| 0525-F6 | No bibkey lint | `[RESOLVED]` | `scripts/check-bibkeys.mjs` present |
| 0525-F7 | No theorem xref-label lint | `[RESOLVED]` | `scripts/check-xref-labels.mjs` present |
| 0525-F8 | Julia + torch companion-rigor gap | `[PARTIAL]` | Julia rigor present (runtests.jl ch04–06, `_shared/JuliaFormatter.toml`); **torch still absent** → see `0527-F27` |
| 0525-F9 | No `.pre-commit-config.yaml` | `[RESOLVED]` | `.pre-commit-config.yaml` present |
| 0525-F10 | No `Makefile` | `[RESOLVED]` | `Makefile` present |
| 0525-F11 | No `docs/STATUS.md` | `[RESOLVED]` | `docs/STATUS.md` + `scripts/generate-status.mjs` present |
| 0525-F12 | LICENSE absent | `[RESOLVED]` | `LICENSE` present (ahead of Track-C "defer until maturity") |
| 0525-F13 | CONTRIBUTING.md absent | `[RESOLVED]` | `CONTRIBUTING.md` present |
| 0525-F14 | No CI workflow | `[RESOLVED]` | `.github/workflows/{deploy,validate}.yml` present |
| 0525-F15 | No issue/PR templates | `[RESOLVED]` | `.github/ISSUE_TEMPLATE/{companion,content}_issue.yml`, `config.yml`, `pull_request_template.md` present |
| 0525-F16 | No `.editorconfig` | `[RESOLVED]` | `.editorconfig` present |
| 0525-F17 | Practice-tags MDX equivalent | `[RESOLVED]` | `src/components/Tag.astro` present (substantive use still gated on pilot output — by design) |
| 0525-F18 | `precision.md` adoption | `[WITHDRAWN]` | unchanged (misclassification; see inaugural §4 F18) |
| 0525-F19 | Ch6 energy-drift magnitude claims | `[RESOLVED]` | `ch06:...energy_drift.png` caption reconciled to Δ=0.3 (~10⁻²) + Δ=0.05 (~10⁻⁶) horizon-invariance framing |

**Inaugural net:** 15 RESOLVED, 2 PARTIAL (fold forward), 1 WITHDRAWN. The standards/infra debt the inaugural audit raised is essentially closed.

### 1.2 First-deploy audit (`0526`) open items

| Item | Status | Evidence |
|---|---|---|
| `ci:validate` → `prevalidate` migration | `[RESOLVED]` | `package.json:9` has `prevalidate`; no `ci:validate` script (commit `5176ef8`) |
| ssm CLAUDE.md deploy-paradigm update | `[RESOLVED]` | CLAUDE.md "Deploy paradigm" section present (commit `47febc0`) |
| `docs/STATUS.md` staleness watch | `[OPEN]` (not yet stale) | `docs/STATUS.md:3` Verified 2026-05-26 = 3 days old (< 14-day gate) |
| Layer-2 chapter-route cleanup (DML) | `[EXTERNAL]` | `double_ml_time_series` repo |
| `projects.json whats_next` update | `[EXTERNAL]` | `brandon-behring.dev` repo |
| Orphan bare-name Worker cleanup | `[EXTERNAL]` | Cloudflare dashboard (manual) |
| Scaffold validator error-UX issue | `[EXTERNAL]` | `book-scaffold-astro` issue |

### 1.3 Deep audit (`0527-F0` … `0527-F37`)

| ID | Finding (short) | Track | Status | Evidence |
|---|---|---|---|---|
| 0527-F0 | Audit-format drift (0526 + draft use A-numbers) | A | `[OPEN]` | addressed structurally by this pass (§6 `0529-F2`); the 0526/draft docs themselves unchanged |
| 0527-F1 | README chapter-status stale | A | `[OPEN]` | `README.md:5` "Ch 1–3 active; Ch 4–17 stubbed" |
| 0527-F2 | Ch5:96 "5 conditions for order 4" wrong | A | `[OPEN]` | verbatim at `ch05-stability-regions.mdx:96` |
| 0527-F3 | Ch5 Ex 5.1 mislabel | A | `[OPEN]` | verbatim at `ch05:196` ("four order-conditions of order 4" lists order-1/2/3 conditions) |
| 0527-F4 | XRef machinery dead (0 usages) | B | `[OPEN]` | `grep -rc '<XRef'` → 0; 14 IDs, 16 `<Figure>` (2 w/o id) |
| 0527-F5 | Notebook policy undocumented | C | `[DEFERRED-C]` | no `.ipynb`; no policy doc |
| 0527-F6 | Mamba-3 missing from bib + 13 uncited | A | `[OPEN]` | no `mamba3`/`2603.15569` entry in `bibliography.bib` |
| 0527-F7 | Ch4 Julia needs `Pkg.instantiate()`; make excludes ch04 | B | `[OPEN]` | `Makefile:79` loop = `ch05 ch06` |
| 0527-F8 | Scaffold version docs stale | A | `[OPEN]` | now WORSE — installed `^4.8.0`, CLAUDE.md says v4.2.0 → `0529-F4` |
| 0527-F9 | Two unused committed figures | A | `[PARTIAL]` | `ch06/stiff_blowup.png` now referenced; **`ch01/matrix_exponential_convergence.png` still unused** |
| 0527-F10 | Astro/KaTeX minor upgrades available | C | `[DEFERRED-C]` | `astro ^6.1.7`, `katex ^0.16.11` (floors) |
| 0527-F11 | CLAUDE.md:22 says exercises in separate files | A | `[OPEN]` | exercises embedded (`## 5.7 Exercises`); CLAUDE.md:22 unchanged |
| 0527-F12 | Jordan-block dismissal inconsistency | A | `[OPEN-STALE-REF]` | claim now at `ch01:121` (was :111) + `ch03:71`; +10 drift from `966b619` |
| 0527-F13 | Ch2 BIBO pole-zero caveat tension | C | `[DEFERRED-C]` | statement mathematically correct |
| 0527-F14 | HiPPO-LegS κ-bounded claim overstated | A | `[OPEN]` | `ch03-linear-algebra.mdx:116` + Fig 3.1 caption |
| 0527-F15 | Ch4 ZOH C¹ regularity unstated | C | `[DEFERRED-C]` | `ch04:114` |
| 0527-F16 | Gauss-Legendre "unique" under-qualified | A | `[DISPUTED]` | `ch06:135`; see §4 — "unique s-stage RK of order 2s" is defensible (2s is max order, GL the unique attainer); "implicit" is a clarifying nicety, not a correction |
| 0527-F17 | Ch6 Hamiltonian C¹ assumption unstated | C | `[DEFERRED-C]` | `ch06:103` |
| 0527-F18 | Gauss-Legendre symplecticness cite missing | A | `[OPEN]` | no `hairer2006geometric`/Geometric-Integration entry in bib |
| 0527-F19 | `gu2024mamba` wrong BibTeX type | A | `[OPEN]` | `bibliography.bib:134` `@article{gu2024mamba` |
| 0527-F20 | `dao2024mamba2` wrong BibTeX type | A | `[OPEN]` | `bibliography.bib:142` `@article{dao2024mamba2` |
| 0527-F21 | `anonymous2025lyapunov` metadata speculative | C | `[DEFERRED-C]` | `bibliography.bib:151` `author={Anonymous}` |
| 0527-F22 | No `confirmed_review` markers | C | `[DEFERRED-C]` | `grep confirmed_review` → 0 |
| 0527-F23 | 4 cited-nowhere bib entries | C | `[DEFERRED-C]` (info) | forward refs to Ch 7+ |
| 0527-F24 | ~259 refs missing vs post_transformers | C | `[DEFERRED-C]` | Ch 7+ forward refs |
| 0527-F25 | Flat bib vs dossier organization | C | `[DEFERRED-C]` | recommend before Ch 10 |
| 0527-F26 | JAX companions: no pytest | B | `[OPEN]` | `find companions -path '*/jax/*' -name 'test_*.py'` → 0 |
| 0527-F27 | Torch parity gap | B | `[OPEN]` | ch01–03 torch = empty `.gitkeep`; ch04–06 torch dirs absent |
| 0527-F28 | Docstring coverage gaps (ch05–06) | A | `[OPEN]` | unchanged (audit table per-file) |
| 0527-F29 | **Julia exp-trap math bug** | A | `[RESOLVED]` | commit `08f7e7c`: `discretization_atlas.jl:99` `T(1.0)`→`dt` + derivation comment |
| 0527-F30 | JAX exp-trap docstring shows `1` not `dt` | A | `[OPEN]` | `exp_trapezoidal.py:78` LaTeX shows `1`; code `:113` sets `dt` |
| 0527-F31 | Ch4 caption k=2 vs code k=4 | A | `[OPEN-STALE-REF]` | `_K_STIFF=4.0` confirmed (`discretization_comparison.py:74`); cited caption `ch04:195` has changed content — re-locate the k=2 caption before fixing |
| 0527-F32 | Ch6 stiff_demo μ=30 docstring vs μ=10 code | A | `[OPEN]` | `stiff_demo.py:11` docstring μ=30; `:47` `_MU=10.0`; figure title uses `_MU` (=10) |
| 0527-F33 | Ch4 Julia missing exp-trap test | A | `[RESOLVED]` | commit `08f7e7c` added `@testset "Exp-trapezoidal achieves order >= 2"` (10/10 pass) |
| 0527-F34 | Ex 6.3 cites unreproducible Δ=0.05 drift | A | `[OPEN]` | `ch06:198`; `symplectic_demo.py main()` runs Δ=0.3 |
| 0527-F35 | Ruff I001 in `lyapunov_qr.py` | A | `[OPEN]` | `lyapunov_qr.py:46` `import sys` after third-party; `# noqa: E402` only |
| 0527-F36 | Makefile help vs loop mismatch | A | `[OPEN]` | help `:27` says ch04/05/06; loop `:79` = ch05/06 |
| 0527-F37 | Deps minimum-only pins | C | `[DEFERRED-C]` | `_shared/pyproject.toml` |

**Deep-audit net:** 2 RESOLVED (`F29`, `F33`), 2 PARTIAL-ish (`F8`→worse, `F9`), 1 DISPUTED (`F16`), 9 DEFERRED-C, the remainder OPEN.

### 1.4 Draft content-quality audit (`0527` draft, A1–A10)

`[SUPERSEDED]` in full — reconciled into deep-audit F-numbers per its §A. No independent status.

---

## 2. Resolved since the audits were written (evidence)

- **`08f7e7c`** (2026-05-28) — closes `0527-F29` **and** `0527-F33`: `companions/ch04/julia/discretization_atlas.jl:99` now sets `dt` (with a derivation comment citing `exp_trapezoidal.py:105-109`), and `runtests.jl` gained an empirical convergence regression test (dt ∈ {0.2, 0.1, 0.05}, threshold 0.35). This was the single highest-severity, pilot-blocking finding in the deep audit — it is no longer open.
- **`5176ef8`** (2026-05-27) — closes the first-deploy `ci:validate` migration: `package.json` now uses the `prevalidate` lifecycle hook; `validate-command` reverted to `validate`.
- **`47febc0`** (2026-05-26) — closes the first-deploy "CLAUDE.md deploy-paradigm" item.
- **Inaugural infra slate** (`0525-F5`–`F16`, `F17`, `F19`) — all the named files now exist (`STYLE.md`, `scripts/{check-bibkeys,check-xref-labels,generate-status}.mjs`, `Makefile`, `.pre-commit-config.yaml`, `docs/STATUS.md`, `LICENSE`, `CONTRIBUTING.md`, `.editorconfig`, `.github/workflows/`, `.github/ISSUE_TEMPLATE/`, `src/components/Tag.astro`). The Track-C governance debt closed *ahead of* the inaugural "defer until maturity" schedule.

---

## 3. Still-open CONTENT / CITATION findings (flag-only — NOT edited)

Per the scope of this pass, each is flagged with location, proposed fix, and my confidence. The author adjudicates; nothing here was changed.

- **`0527-F1` — README status stale.** `README.md:5` ("Ch 1–3 active; Ch 4–17 stubbed"); `README.md:48` ("v4.2.0"). *Proposed:* "Ch 1–6 implemented; Ch 7–17 planned" + bump version string. **Confidence: HIGH.** Truthfulness; not pilot-blocking.

- **`0527-F2` — Ch5 order-condition count.** `ch05:96` "5 conditions for order 4, 17 for order 5, 37 for order 6." The 17/37 are the *cumulative* Butcher-tree counts (≤5 nodes = 17, ≤6 = 37); cumulative order-4 is **8**, not 5 (exact-order-4 is 4). *Proposed:* per audit, restate as cumulative "1, 2, 4, 8, 17, 37 for p=1..6." **Confidence: HIGH** (internal inconsistency with the 17/37 sequence is arithmetic). Pilot-adjacent (foundational).

- **`0527-F3` — Ch5 Ex 5.1 mislabel.** `ch05:196` asks to verify "the four order-conditions of order 4" but lists ∑b=1, ∑b·c=1/2, ∑b·c²=1/3 — the orders 1/2/3 conditions. *Proposed:* either reframe as "conditions through order 3" or add the genuine four order-4 tree conditions. **Confidence: HIGH.** Pilot-adjacent.

- **`0527-F6` — Mamba-3 citation gap.** No `bibliography.bib` entry; ~13 uncited mentions (ch02:169, ch03:118, ch04:33/52/176/197/203, ch06:44/46/74/85, +2 ch05). *Proposed:* add `lahoti2026mamba3` (arXiv:2603.15569, ICLR 2026) + `<Cite>` at each site; for the A-stability claim additionally cite an exponential-integrator source. **Confidence: HIGH** (entry absence verified). **Pilot-relevant** (C1 anchors on these chapters).

- **`0527-F11` — CLAUDE.md exercises-location.** `CLAUDE.md:22` says exercises live in `chXX/exercises.mdx`; they are embedded (`## 5.7 Exercises`, `## 6.8 Exercises`). *Proposed:* correct CLAUDE.md:22 to "embedded in chapter MDX." **Confidence: HIGH.** (This is the claim that produced the explorer false-negative — see `0529-F5`.)

- **`0527-F12` — Jordan-block dismissal.** `ch01:121` + `ch03:71` call non-diagonalizability "essentially always" negligible, yet ch01 §1.3 uses critical damping (a Jordan case) as canonical, and ch02's Lyapunov theorem carves out defective blocks. *Proposed:* soften to "generic dense random A → measure-zero; structured/learned A (HiPPO-LegS, S4 DPLR, critical damping) may be Jordan by design." **Confidence: MEDIUM** (internal inconsistency is real; the exact rewrite is a pedagogical judgment call). Pilot-relevant.

- **`0527-F14` — HiPPO-LegS κ claim.** `ch03:116` + Fig 3.1 caption: "condition number bounded as N grows." Research-kb (arXiv:2310.01698 Thm 5) indicates sub-quadratic *growth*, not bounded. *Proposed:* add a `<Cite>` and soften "bounded" → "grows slowly (sub-quadratically)." **Confidence: MEDIUM** (depends on which κ and matrix the source measures vs. the chapter's claim — author should check against the HiPPO paper directly). Pilot-relevant.

- **`0527-F16` — Gauss-Legendre "unique."** `ch06:135` "the unique s-stage RK method of order 2s." `[DISPUTED]`: 2s is the maximum attainable order for an s-stage RK method and Gauss-Legendre is its unique attainer (Butcher), so the statement is defensible without the "implicit" qualifier. Adding "implicit" (or "collocation") is a clarifying nicety, not a correction. *Proposed:* optional clarification only. **Confidence the audit over-flagged: MEDIUM-HIGH.** Author should confirm before any edit.

- **`0527-F18` — Gauss-Legendre symplecticness cite.** `ch06:135` asserts symplecticness; §6.4 cites HLW Vol II (the *stiff* volume), not the *Geometric Numerical Integration* volume where the result lives. *Proposed:* add `hairer2006geometric` (and optionally Sanz-Serna 1988). **Confidence: MEDIUM-HIGH** (citation-discipline; the assertion itself is correct). Pilot-relevant.

- **`0527-F19` / `0527-F20` — BibTeX type errors.** `bibliography.bib:134` `@article{gu2024mamba` (COLM) and `:142` `@article{dao2024mamba2` (ICML) are conference papers. *Proposed:* `@inproceedings` + `booktitle=`; regenerate `references.json`. **Confidence: HIGH** (COLM/ICML are conferences).

- **`0527-F30` — JAX exp-trap docstring.** `exp_trapezoidal.py:78` LaTeX shows `1` in the augmented-matrix off-diagonal; code `:113` sets `dt`. (Same class as the now-fixed `0527-F29`, but in the docstring.) *Proposed:* update the LaTeX to `dt`. **Confidence: HIGH.**

- **`0527-F31` — Ch4 caption k mismatch.** `[OPEN-STALE-REF]`: `_K_STIFF=4.0` confirmed in `discretization_comparison.py:74`, but the cited caption `ch04:195` now describes `order_convergence.png` (different content). *Proposed:* re-locate the `k=2` caption (likely moved) before deciding caption-vs-code reconciliation. **Confidence the code uses k=4: HIGH; that a live caption still says k=2: UNVERIFIED.**

- **`0527-F32` — Ch6 μ mismatch.** `stiff_demo.py:11` docstring + `ch06:262` prose say μ=30; code `:47` `_MU=10.0` (and the figure title renders μ=10). *Proposed:* pick one value and reconcile docstring + prose + code. **Confidence: HIGH.**

- **`0527-F34` — Ex 6.3 unreproducible drift.** `ch06:198` cites a Δ=0.05 drift rate; `symplectic_demo.py main()` runs only Δ=0.3. *Proposed:* add a Δ=0.05 run or re-cite Δ=0.3. **Confidence: HIGH** (the cited number is not script-reproducible).

- **`0527-F4` — XRef machinery (architectural).** 0 `<XRef>` usages; 14 theorem IDs orphaned; 2 of 16 figures lack `id=`. *Proposed:* decide Option A (enforce: add figure IDs + introduce `<XRef>` + flip `check-xref-labels.mjs` to fail) vs Option B (retract the docs claim). **Confidence: HIGH** (state verified); the decision is yours.

---

## 4. Still-open INFRA / TOOLING / DOC-DRIFT findings

- **`0529-F4` / `0527-F8` — scaffold-version doc-drift (now 6 versions stale).** `CLAUDE.md:3` + `:108` say "v4.2.0"; installed is `^4.8.0`. `CLAUDE.md:69` still documents `npm run ci:validate` (script removed; now `validate` + `prevalidate` hook). `README.md:48` says "v4.2.0." *Proposed:* update to v4.8.0 (or version-agnostic) + fix the ci:validate line. **Confidence: HIGH.**
- **`0527-F26` — JAX companions have no pytest** (0 `test_*.py`). Julia has `runtests.jl`; the asymmetry is exactly what let `0527-F29` survive. Track B.
- **`0527-F27` — torch parity.** ch01–03 `torch/.gitkeep` (empty advertised dirs); ch04–06 have no torch dir. *Proposed:* delete the `.gitkeep`s + document deferral, or populate. Track B.
- **`0527-F28` — docstring coverage** (ch05/06 JAX 27–50%, ch06 Julia 0–20%). Track A maintenance.
- **`0527-F35` — ruff I001** in `lyapunov_qr.py:46`. Track A.
- **`0527-F36` / `0527-F7` — Makefile help (`:27`) vs loop (`:79`)**: help advertises ch04/05/06; loop runs ch05/06 (ch04 excluded pending `Pkg.instantiate()`). Track A doc fix.
- **`0527-F9` (partial) — unused figure** `public/figures/ch01/matrix_exponential_convergence.png` (0 references). `stiff_blowup.png` is now used.
- **Manifest.toml policy** — `companions/ch04/julia/Manifest.toml` is untracked **and not gitignored** (no `.gitignore` rule); ch05/ch06 have none. *Proposed:* decide commit-for-reproducibility vs add a `.gitignore` rule, and apply consistently across ch04–06. **Confidence the inconsistency exists: HIGH.**
- **Stale remote branch** `origin/update_worker_name_to_ssm-foundations` — work integrated; safe to delete. Hygiene.

---

## 5. Pilot-imminent re-gate (as of 2026-05-29)

C1 pilot ≈ 2026-06-01 (~3 days), anchored on Ch 1–3 + Ch 6. Re-evaluating the deep-audit's promoted set against `HEAD`:

- **Already shipped:** `0527-F29`, `0527-F33` (the only code-correctness blockers).
- **Still open and genuinely pilot-relevant:** `0527-F6` (Mamba-3 citations across the anchored chapters), `0527-F14` (Ch3 HiPPO κ), `0527-F18` (Ch6 symplectic cite), `0527-F12` (Ch1 Jordan framing). All are content/citation flags in §3 — author-adjudicated, not code blockers.
- **Reclassified:** `0527-F16` `[DISPUTED]` — likely not an error; should not be treated as pilot-blocking.

Net: **no remaining code-level pilot blocker.** The pilot-relevant residue is prose/citation polish in the anchored chapters, all itemized in §3.

---

## 6. New findings this pass (`0529-Fn`) + forward registry

- **`0529-F1` — provenance backfill gap.** v4.8.0 ships a per-chapter provenance block; only ch01 was backfilled (commit `966b619`). ch02–06 (all `implemented`) have 0 provenance markers. *Proposed:* backfill ch02–06 or document the deferral. Track A. **Confidence: HIGH.**
- **`0529-F2` — F-number namespace collision (fixed here).** `0525-Fn` and `0527-Fn` are independent namespaces (`0525-F8` ≠ `0527-F8`). **Forward convention (registry):** all findings are referenced `YYYYMMDD-Fn` by their originating audit's date; bare `Fn` in commit messages (e.g. `08f7e7c` "F29") resolves to the contemporaneous audit (`0527-F29`). Future audits date-prefix. This extends the `project_ssm_foundations_audit_format` memory.
- **`0529-F3` — audit line-ref rot.** `966b619` added 10 lines to ch01 → `0527-F12` ch01:111→121; `0527-F31`'s cited caption (`ch04:195`) shifted content. Future audits should cite stable anchors (theorem IDs / nearest heading), not raw line numbers, or be re-anchored on each content edit.
- **`0529-F4` — doc-drift compounded.** See §4 (CLAUDE.md/README v4.2.0 vs installed v4.8.0; ci:validate doc remnant).
- **`0529-F5` — process note (explorer false-negative).** A parallel exploration of this very repo concluded "zero exercises exist" by grepping `## Exercises` / `exercises.mdx`, missing the numbered headings (`## 5.7 Exercises`). Future audit prompts must search for `## N.M Exercises` and `### Exercise N.M`. No repo change; method caution.

---

## 7. Recommended next actions (NOT executed — report-only)

Prioritized; the author chooses what to pick up.

1. **Doc-drift sweep (Track A, trivial, high-truthfulness):** `0527-F1`, `0529-F4`, `0527-F11`, `0527-F36`. One commit; no math judgment.
2. **Citation discipline (Track A, pilot-adjacent):** `0527-F6` (Mamba-3 entry + cites), `0527-F19`/`F20` (bib types), `0527-F18` (symplectic cite). Mechanical once the entries are chosen.
3. **Content correctness (author-adjudicated):** `0527-F2`, `0527-F3` (HIGH-confidence math), then `0527-F12`, `0527-F14` (judgment calls). Confirm `0527-F16` is a non-issue before touching it.
4. **Code-prose parity (Track A):** `0527-F30`, `0527-F32`, `0527-F34`, `0527-F35`; re-locate `0527-F31` first.
5. **Provenance + hygiene:** `0529-F1` (backfill ch02–06), Manifest.toml policy, delete the stale remote branch.
6. **Track B (systemic):** `0527-F4` (XRef enforce-vs-retract decision), `0527-F26` (JAX pytest), `0527-F27` (torch).
7. **Track C (unchanged, intentional):** `0527-F5`, `F10`, `F13`, `F15`, `F17`, `F21`–`F25`, `F37`.

---

## 8. Verification provenance

All statuses determined read-only from `/home/brandon_behring/Claude/ssm-foundations` on 2026-05-29, HEAD `966b619`. Key commands:

```bash
git show --stat 08f7e7c                                   # 0527-F29/F33 RESOLVED
ls scripts/ .github/workflows/ .github/ISSUE_TEMPLATE/ \
   LICENSE CONTRIBUTING.md .editorconfig STYLE.md         # 0525 infra RESOLVED
grep -niE "mamba3|2603.15569" bibliography.bib            # empty → 0527-F6 OPEN
grep -nE "@article\{(gu2024mamba|dao2024mamba2)" bibliography.bib  # 0527-F19/F20 OPEN
grep -n "5 conditions for order 4" src/content/chapters/ch05-stability-regions.mdx  # 0527-F2 OPEN
sed -n '196p' src/content/chapters/ch05-stability-regions.mdx     # 0527-F3 OPEN
grep -rc '<XRef' src/content/chapters/*.mdx               # 0 → 0527-F4 OPEN
grep -ci provenance src/content/chapters/ch0{2..6}-*.mdx  # 0 each → 0529-F1
grep -n "_MU" companions/ch06/jax/stiff_demo.py           # 0527-F32 OPEN (μ=10 vs docstring 30)
git check-ignore companions/ch04/julia/Manifest.toml      # not ignored → policy gap
git branch -r                                             # stale update_worker_name branch
```

**Read-only guarantee:** this pass created exactly one file (this document). `git diff` is empty; the three prior audits, all source/config/prose, and git history are untouched. The pre-existing untracked `companions/ch04/julia/Manifest.toml` is unchanged.
