# ssm-foundations — Post-M6 Recheck Audit (Stage 1 beta gate, Phase A)

**Audit date:** 2026-06-13
**Scope:** Ch 1–10 — the chapters authored *before* the `claim-skeptic` discipline existed (added 2026-06-04). Stage 1 of the post-M6 beta gate: bring Ch 1–10 to the Ch 11–17 quality bar.
**Method:**
- **A1** — re-verify the 2026-05-27 deeper audit (37 F-findings) against the *current* state (read-only Explore fan-out, 3 agents).
- **A2** — first-ever `claim-skeptic` adversarial math-truth sweep over Ch 1–10 (10-agent Workflow, run `wf_647b37ff-ec5`; 855K subagent tokens). Each agent refuted-by-default one chapter's theorem statements, derivations, definitions, attributions, and numeric claims vs its own derivations + cited sources + committed companion outputs.
- **A3** — this merged, deduped, triaged backlog. The Ch 1–6 high-stakes findings were re-verified against the source files before entry here (the methodology-audit grounding rule).
**Supersedes/extends:** `audits/2026-05-27_repo_audit_deeper.md` (A1 reconciles its F-numbers).
**Format:** house audit format (`project_ssm_foundations_audit_format.md`) — severity · Track A/B/C · bracketed status. New findings carry recheck IDs **R##**, each cross-referencing its origin (a 2026-05-27 F-id, or an A2 `chXX:S#`).
**Artifact policy:** findings-only; remediation in Phase B PRs.

---

## Executive summary

- **A1 result — the 2026-05-27 audit is mostly already closed.** 30 of 37 findings ALREADY-FIXED by the 2026-06-04 runway + later passes — **including every CRITICAL**: F1 (README), F2/F3 (Ch 5 RK counts + Ex 5.1), F4 (XRef forward-only policy in STYLE §4; Ch 1–6 figures carry `id=`), F6 (Mamba-3 bib + cites), F18 (`hairer2006geometric` cite present), F19/F20 (bib `@inproceedings`), F26 (Ch 1–3 JAX pytest), F27 (torch parity ch01–06), **F29 (the Julia exp-trap *math bug* — now `=dt` with an F29 regression `@testset`)**, F30/F31/F32/F33/F34/F35. **7 residuals remain:** F16, F14, F7, F9, F36, F15, F17.
- **A2 result — the never-run net caught real errors.** 42 raw findings (all 10 chapters). After triage + artifact re-verification: **11 must-fix** (genuine math errors, internal contradictions, and wrong/over-specified citation keys), **18 should-fix** (imprecision + the "caption claims a number/check no artifact produces" provenance pattern), **11 optional/minor** (uncertain or stylistic), **2 confirmed-correct** (no action). The agent was highly reliable — every Ch 1–6 refuted finding re-read against source held up.
- **Combined actionable backlog: 12 must-fix + 23 should-fix.** (A2's 11 must-fix + A1's F16; A1's F14 == A2 ch03:S3. A2's 18 should-fix + A1's F7/F36/F15/F17/F9.)
- **Most consequential new findings:** **ch03:S1** (a *false* Jordan-block-size formula in the Jordan-form theorem, contradicted by the chapter's own Ex 3.1) · **ch04:S1** (a self-contradictory ZOH-vs-bilinear modulus claim, load-bearing for the Ch 6 symplectic motivation) · **ch05:S1** (the trilemma + NoteBox contradict the chapter's own Theorem 5.3) · **ch07:S1/S2** (conditioning-growth misattributed to `park2024numerical`, which proves no such thing — the sub-quadratic result is `yu2023robustifying`'s, and for a *different* matrix) · three wrong/over-specified citation keys (**ch03** Longhorn→`dao2024mamba2`, **ch06:106** HLW→`hairer1996ordinary`).
- **Pattern echoes prior lessons:** the ch06:S5/S6 + ch08:S3 + ch09:S1 + ch10:S5 "prose/caption cites a magnitude that no committed test pins" is the ch15/F34 provenance lesson, recurring across the early chapters.

---

## Stage 2 resolution (2026-06-13, PR A — tooling/CI)

The tooling/CI residuals are closed in one PR (no chapter-content change):

- **#26 — content-validating `status-check`.** `scripts/generate-status.mjs --check` now regenerates the snapshot from chapter frontmatter and byte-compares it to `docs/STATUS.md` (modulo the verified date), keeping the ≤14-day staleness check. A `node --test` suite (`scripts/generate-status.test.mjs`, wired into `make check` via the new `test-scripts` target) guards the logic. **On its first run it caught a real Stage-1 drift** the old date-only check missed: ch03 `Cites` 7→9 (the R3 DeltaNet/Longhorn cites), ch06 5→4 (the R6 `hairer2006geometric` cite-key swap), ch10 `Lines` 449→450 — `docs/STATUS.md` regenerated to match.
- **F7/#4 — ch04 Julia folded into the default loop.** `Makefile` `companion-julia-tests` now runs `ch04 ch05 …`; the comment + help text document the one-time `Pkg.instantiate()` (DifferentialEquations.jl). Verified locally: ch04 10/10 green, full julia loop green. **F36 subsumed** — the help text is now ch04-inclusive.
- **F26 — confirmed.** JAX pytest assert suites are real across ch01–10 (40 tests in ch01–03 alone; `companion-verifier` clean).
- **F27 — completed for ch01–03.** A1 recorded "torch parity ch01–06", but ch01–03 only validated against analytic/library oracles (scipy `expm`, `torch.linalg.matrix_exp`) — not a literal torch↔jax cross-check. Stage 2 backfilled `companions/ch0{1,2,3}/torch/tests/test_*_parity.py` (9 tests; `companion-verifier` confirmed genuine cross-framework, no stubs). Torch↔jax parity is now real ch01–16.

F7 flips to `[done]` below. (F9/F15/F17/F36 were closed in Stage 1 PR #36; F16/F14 and the R-numbered findings shipped in PRs #32–#36.)

---

## A1 — 2026-05-27 audit residuals (the 7 still-open)

| F# | Sev | Track | Site | Residual | Status |
|---|---|---|---|---|---|
| F16 | IMPORTANT | A (pilot) | `ch06:148` | "the *unique* $s$-stage RK method of order $2s$" — missing **implicit** qualifier | `[open]` |
| F14 | IMPORTANT | A (pilot) | `ch03:127` | HiPPO-LegS κ "∼N²" has no `<Cite>`; matrix-κ vs eigenvector-basis-κ not distinguished (= A2 ch03:S3) | `[open]` |
| F7 | IMPORTANT | B | `Makefile`/`companions/ch04/julia` | ch04 Julia excluded from default `companion-julia-tests`; `Manifest.toml` now exists → can fold in (old #4) | `[done — Stage 2 PR A]` |
| F9 | MINOR | A | `public/figures/ch01/matrix_exponential_convergence.png` | orphaned (unused). NB `ch06/stiff_blowup.png` **is** used — audit was wrong on that one | `[open]` |
| F36 | MINOR | A | `Makefile:27` | help text "ch05, ch06, ch07" ≠ actual loop (ch05/06/07/10/11/12/13/15/17) | `[open]` |
| F15 | MINOR | A | `ch04:185`-area | exp-trap "$u$ is $C^1$" + ZOH "smooth $u$" — regularity hypothesis loose (see R-regularity cluster) | `[open]` |
| F17 | MINOR | A | `ch06:116` | energy-conservation derivation assumes but doesn't state $H\in C^1$ | `[open]` |

*All other 2026-05-27 F-findings: `[fixed]` (verified against current state) or Track-C deferred (F5/F10/F13/F21–F25/F37, unchanged).*

---

## A2 — claim-skeptic backlog (triaged)

### MUST-FIX — genuine errors / false statements / wrong citation keys

| R# | Origin | Sev | Site | Problem (verified ✓ = re-read against source) | Track |
|---|---|---|---|---|---|
| R1 | ch02:S1 | High* | `ch02:296` | Lyapunov-uniqueness reason **wrong**: kernel trivial "when $\statemat$ has no purely imaginary eigenvalues" — should be "no two eigenvalues sum to zero" ($\lambda_i+\lambda_j\neq0$). Counterex `diag(1,-1)`. Conclusion holds, stated reason false. ✓ | A (pilot) |
| R2 | ch03:S1 | **High** | `ch03:53` | **False math** in the Jordan-form theorem: "size of each Jordan block = algebraic − geometric multiplicity". Correct: #blocks = geometric mult, Σ block sizes = algebraic mult. Chapter's own Ex 3.1 ($J_2(2)$) refutes it. ✓ | A (pilot) |
| R3 | ch03:S2 | **High** | `ch03:226` | Wrong cite key + wrong attribution: "The Longhorn paper `<Cite key="dao2024mamba2"/>`" — `dao2024mamba2` is Mamba-2; Longhorn is `liu2024longhorn`. And the displayed explicit-Euler update is **DeltaNet**, not Longhorn (the implicit/backward-Euler cousin — see ch12:174–207). ✓ | A (pilot) |
| R4 | ch04:S1 | **High** | `ch04:157` | Self-contradictory + false: "ZOH crushes [imaginary modes] to the unit-disk interior … never exactly on the circle" — but $\lvert e^{i\omega\Delta}\rvert=1$ exactly on the circle, and ZOH is autonomous-*exact*. Real ZOH↔bilinear distinction is aliasing/phase-warp, not interior-vs-circle. Load-bearing for the Ch 6 symplectic motivation. ✓ | A (pilot) |
| R5 | ch05:S1 | **Important** | `ch05:42, 177` | NoteBox ("no explicit RK A-stable *and* order > 2") + trilemma ("at most two of {explicit, A-stable, order≥3}") **contradict the chapter's own Theorem 5.3** ("explicit methods cannot achieve A-stability at *any* order"). The order qualifier is the Dahlquist *second barrier* for A-stable **linear multistep** methods, mis-transplanted onto explicit RK. ✓ | A (pilot) |
| R6 | ch06:S1 | **Important** | `ch06:106` | Wrong cite key: "Hairer–Lubich–Wanner `<Cite key="hairer1996ordinary"/>` Volume II Ch VI" — `hairer1996ordinary` is the **2-author** stiff-ODE vol (Hairer & Wanner); the 3-author HLW *Geometric Numerical Integration* is `hairer2006geometric` (correctly used at :148). ✓ | A (pilot) |
| R7 | ch06:S3 | **Important** | `ch06:153` | Boxed Theorem (modified Hamiltonian) **missing hypotheses**: the exp-small/exp-long estimate needs $H$ **real-analytic** + step size below a problem-dependent $\Delta_0$; as stated it claims this for any Hamiltonian + any $\Delta$. "exactly preserves $\widetilde H$ to exp-small terms" should be the *optimally-truncated* $\widetilde H_N$ (the full series diverges). Proof sketch :156 already says "truncated with controlled remainder". ✓ | A (pilot) |
| R8 | ch07:S1+S2 | **High** | `ch07:122, 124` | **Misattribution + wrong rate.** "condition number … sub-quadratically `<Cite key="park2024numerical"/>`" — park2024 proves LegS-ODE well-posedness + discretization *convergence*, **not** conditioning growth. Sub-quadratic ($n^{3/2}$) is `yu2023robustifying` Thm 5, and for the **perturbed** transform $\tilde V$; the **raw** HiPPO eigenvector matrix is *exponentially* ill-conditioned (yu2023 p.7). ⚠ verify vs `research-kb/fixtures/papers/arxiv/2412.08595.pdf` at fix-time | A (pilot) |
| R9 | ch10:S1 | **Important** | `ch10:64, 86` | Internal contradiction: §10.2 scheme identified with the §4.5 exponential-trapezoidal scheme, but the chapter's own MarginNote (:105–110) says they are **different** schemes (φ-family interpolation vs trapezoidal *quadrature*; differ at $O(\Delta^3)$). ⚠ verify at fix-time | A |
| — | F16 | IMPORTANT | `ch06:148` | (A1) add "implicit" qualifier — see above | A (pilot) |
| — | F14 / ch03:S3 | IMPORTANT | `ch03:127` | (A1=A2) HiPPO-κ cite + matrix-κ vs eigenvector-basis-κ split — see R8's two-object distinction | A (pilot) |

*\*R1 severity: the agent rated it Important; tagged High* here because it is a **false reason in a proof** (a truth defect), though the proven conclusion is unaffected.*

### SHOULD-FIX — imprecision, provenance, wrong-but-not-load-bearing

| R# | Origin | Sev | Site | Problem | Track |
|---|---|---|---|---|---|
| R10 | ch01:S1 | Important | `ch01:155` | Caption "decay rate of $c/2$ in the energy norm" but the figure plots energy $E$ (not $\sqrt E$), whose log-slope is $-c$; companion labels envelope $e^{-ct}$, test pins slope ≈ $-c$. Conflates the two. | A (pilot) |
| R11 | ch01:S2 | Moderate | `ch01:200` | "three of these [six behaviors]: stable node, **stable degenerate node**, stable spiral" — but the degenerate node is on the boundary $\tr^2=4\det$, **not** one of the six tabulated rows. | A (pilot) |
| R12 | ch04:S2 | Moderate | `ch04:237` | Ex 4.2 solution: "the truncation in $\varphi_1$" — wrong object; bilinear's $\bar A$ is the (1,1)-Padé of $e^z$ (chapter's own MarginNote :159), and $\varphi_1$ governs $\bar B$, not $\bar A$. | A (pilot) |
| R13 | ch04:S3 | Moderate | `discretization_atlas.jl:15,230` + `jax/…:70` + `torch/…:46` + `runtests.jl:42` | Companion comment eigenvalues of $[[0,1],[-4,-0.5]]$ given as $-0.25\pm i\sqrt{15}/4\approx\pm0.968i$; correct is $-0.25\pm i\sqrt{15.75}/2=-0.25\pm i\,3\sqrt7/4\approx\pm1.984i$. Comment-only (numerics use eigvals), but a committed wrong claim in 4 files. | A (pilot) |
| R14 | ch05:S2 | Moderate | `ch05:117` | Caption hard-codes "Theorem 5.2 (Butcher order conditions)"; `labels.json` resolves `ch05:order-conditions`→Thm **5.1** and `ch05:dahlquist-barrier`→5.2. Use `<XRef>` or drop the number (playbook gotcha). | A (pilot) |
| R15 | ch05:S3 | Moderate | `order_verification.py:70` (jax) + `:47` (torch) | Same $\sqrt{15}/4$ eigenvalue error as R13, in 2 ch05 companion comments. | A (pilot) |
| R16 | ch06:S2 | Moderate | `ch06:83` | "Crank–Nicolson … a 2-stage **SDIRK** of order 2" — CN's tableau has an explicit first stage + unequal diagonals; it is **ESDIRK**, violating both the DIRK ("non-zero diagonal") and SDIRK ("equal diagonals") defs stated one sentence earlier. | A (pilot) |
| R17 | ch06:S4 | Moderate | `ch06:124` | "RK4's **symmetric** stability function" — RK4 is **not** symmetric ($R(z)R(-z)=1+z^6/72\neq1$); the slight damping ($\lvert R(i\theta)\rvert^2=1-\theta^6/72<1$) arises *because* it is asymmetric. Drop "symmetric", give the correct mechanism. | A (pilot) |
| R18 | ch06:S5 | Moderate | `ch06:150` | Caption provenance overstated: "growing to ∼$10^{-1}$ at 1000 periods" computed nowhere; "magnitudes verified by `…/julia/runtests.jl`" **false** (runtests only runs Δ=0.05/t=1.5 and disclaims magnitude). | A (pilot) |
| R19 | ch06:S6 | Moderate | `ch06:168` | Caption pendulum "RK4 monotonic drift ∼$1.5\times10^{-5}$ after 50 periods" computed by no committed artifact (phase-portrait fig never computes a drift). Compute+pin or remove. | A (pilot) |
| R20 | ch07:S3 | Moderate | `hippo_matrix.py:36,59` | Docstrings promise a "§7.5 conditioning demo" that **does not exist** in `companions/ch07` (code computes eigen*values*, never an eigenvector-matrix condition number). Undercuts ch07:122 "empirically". | B / A |
| R21 | ch07:S4 | Minor | `ch07:116` | Bilinear attribution: ch07 credits "the original S4 paper"; ch04:143 credits "the S4D paper". Reconcile (S4, S4D, S5 all use bilinear; this book defaults to ZOH). | A |
| R22 | ch08:S1 | Moderate | `ch08:106,112,116` | S4 kernel complexity stated "$O(N\log^2 N)$ overall" drops the $L$ term; honest cost is $\tilde O(N+L)$ (the inverse FFT alone is $O(L\log L)$, acknowledged at :112 but omitted from the total). | A |
| R23 | ch08:S2 | Moderate | `ch08:110` | Resolvent vs truncated-polynomial: $\hat K$ as a degree-$(L-1)$ polynomial vs $(I-\bar A z)^{-1}$ (infinite series) — the inverse-FFT recovery is exact only for the truncation; the load-bearing $\bar A^L$ correction term is unmentioned. (Section is flagged a non-implemented sketch.) | A |
| R24 | ch09:S1 | Moderate | `ch09:106,108,136,154` | Residual magnitudes ∼$10^{-15}/10^{-13}/10^{-14}$ stated as measured facts in prose + 2 captions, but tests pin only `< 1e-12`. Tighten the asserts or soften the prose (the measured+pinned pattern). | A |
| R25 | ch10:S2 | Moderate | `ch10/jax/discretization.py:170` (+ julia `:72`, torch `:69`) | Docstring "approximates the forcing integral by linear interpolation of the input" but the stated coefficients ($\beta=(1-\lambda)\Delta\alpha,\gamma=\lambda\Delta$) are the **trapezoidal quadrature** form (which §10.2 prose correctly describes), not the φ-family interpolation form. | A |
| R26 | ch10:S3 | Moderate | `ch10:122, 381` | Theorem `ch10:exp-trap-order` hypothesis "$u$ … $C^1$" too weak: the proof bounds the LTE via the trapezoidal-rule $g''$ term, needing $g\in C^2$ ⇒ $u\in C^2$. (ch04 §4.5:185 carries the same loose $C^1$ — part of the regularity cluster with F15/F17.) | A |
| R27 | ch10:S4 | Moderate | `ch10:60` | "changes precisely two things" contradicted by the chapter's own §10.6 ("Two *further* … changes" — MIMO rank-R + production block). Qualify as "two *dynamical-systems* changes". | A |
| — | F7 | IMPORTANT | `Makefile` | (A1) add `companions/ch04/julia` to the default julia-tests loop | B |
| — | F36 | MINOR | `Makefile:27` | (A1) help text mismatch | A |
| — | F15 | MINOR | `ch04:185` | (A1) ZOH/exp-trap regularity (cluster with R26) | A |
| — | F17 | MINOR | `ch06:116` | (A1) state $H\in C^1$ | A |
| — | F9 | MINOR | figure | (A1) delete orphaned `matrix_exponential_convergence.png` | A |

### OPTIONAL / MINOR — uncertain or stylistic (judgment calls; batch or defer)

| R# | Origin | Sev | Site | Note |
|---|---|---|---|---|
| R28 | ch01:S3 | Minor | `ch01:96` | `\norm{M}` used for ≥3 norms (entrywise-$\ell^1$ here, operator elsewhere, Frobenius in Ex 1.5). Optional subscript. |
| R29 | ch02:S2 | Minor | `ch02:114` | "$\mu_i$ are the eigenvalues' magnitudes" then writes $\log\lvert\mu_i\rvert$ (magnitude twice). Reword. |
| R30 | ch02:S3 | Minor | `ch02:138` | "the first to publish" Lyapunov-on-selective-SSMs priority claim (cited to `anonymous2025lyapunov`). Soften to "among the first". |
| R31 | ch02:S4 | Moderate (confirmed) | `ch02:140` | Caption "agreement … validates" overstates *pointwise* match (spectrum degenerate ⇒ QR exponents scatter, only the mean matches tightly). Optional half-sentence (the ch15 degeneracy-scatter lesson). |
| R32 | ch03:S4 | Minor (confirmed) | `ch03:218` | "diagonal plus low-rank — specifically normal plus low-rank" inverts NPLR→DPLR logical order. Optional tightening to match ch08:77. |
| R33 | ch04:S4 | Minor | `ch04:77` | Consistency: "residual vanishes faster than $\Delta$" asserted of the *normalized* $\tau$ (which merely →0). Tighten wording. |
| R34 | ch05:S4 | Moderate | `ch05:175` | Naming: "Dahlquist barrier for explicit RK" + "RK analogue … Butcher" — the no-explicit-RK-A-stability fact isn't standardly the Dahlquist barrier (that's the A-stable-LMM order-≤2 result). Bundle with R5; verify historical attribution. |
| R35 | ch05:S5 | Minor | `ch05:107` | "over 200 order conditions" for 8th-order — cumulative through order 8 is exactly 200. "about 200" / "exactly 200". |
| R36 | ch08:S3 | Minor | `ch08:160` | "∼$3\times10^{-15}$" prefactor not emitted by committed code (only `<1e-12` pinned). Soften or print. |
| R37 | ch10:S5 | Minor | `ch10:132,220` | "∼$2/3\times10^{-15}$" not pinned (only `<1e-12`). Soften or pin. |
| R38 | ch10:S6 | Moderate | `ch10:102,206,260` | Mamba-3 paper attributions (input-dependent λ sigmoid; RoPE on B,C; rank-R MIMO) unverifiable from repo artifacts. **Beta-gate fact-check vs local `lever_of_archimedes/2603.15569.pdf`.** |

### NO-ACTION — confirmed correct (recorded for the trail)

- **ch08:S4** (`ch08:157`) — the associative-scan-second-components = sequential-states claim is correct (inclusive scan, discarded first component); verified vs `s5_scan.py` + test.
- **ch10:S7** (`ch10:164`) — the stiff $A\Delta=-30$ numbers ($\lvert\alpha\rvert\approx9.4\times10^{-14}$ exp, $0.875$ bilinear) are analytically exact ($e^{-30}=9.358\times10^{-14}$; $(1-15)/(1+15)=-0.875$).

---

## Phase B — remediation PR plan

The backlog is larger than the pre-A2 estimate but cleanly themeable. Slicing (each PR through the full gate: `make check-local-torch` + `npm run build` + the touched chapter's review subagents → push → CI → merge → deploy → live check; doc-sync rides IN):

- **B1 — Ch 1–6 math/prose corrections (pilot-critical, ships first).** R1–R7 (must-fix) + R10/R11/R12/R14/R16/R17/R18/R19 (Ch 1–6 should-fix prose) + F16 + F14. The genuine errors C1 builds on. *May split Ch 1–3 / Ch 4–6 if the diff is large.*
- **B1-companions — Ch 1–6 companion fixes** (bundle into B1 or a sibling PR): R13/R15 (eigenvalue comments, 6 files) + F7/#4 (ch04 julia gate) + R26-adjacent regularity. Re-verify with `companion-verifier`.
- **B2 — Ch 7–10 + hygiene.** R8/R9 (must-fix; **verify R8 vs the park2024/yu2023 PDFs first**) + R20–R27 (Ch 7–10 should-fix) + R38 (Mamba-3 fact-check vs local PDF) + hygiene F9/F36/F15/F17 + the R-regularity cluster (F15/F17/R26). *Scope precisely once B1 lands.*
- **B3 — `citation-link-auditor` repo sweep.** Independent cross-check of the cite-key fixes (R3/R6) + bibkey hygiene + cross-repo URL freshness.
- **B4 — STYLE §8 refresh + Stage-1 done-signal.** STYLE §8 (+ §13 Ch 17 note); then remove the `docs/DASHBOARD.md:53/61/63` trust note once B1+B2 land.
- **Optional tier (R28–R37):** fold the cheap ones (R30/R35/R29) into the relevant chapter PR; defer the rest unless a chapter is open anyway.

**Done-signal:** the DASHBOARD "Ch 1–10 do not yet reflect a claim-skeptic review" note is removed once B1+B2 (the must-fix + should-fix) have merged.
