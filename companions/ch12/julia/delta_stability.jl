# delta_stability.jl --- Chapter 12 Julia companion (stdlib only).
#
# Cross-language check of the §12.4 stability analysis for the explicit/implicit
# pair: the delta-rule step (DeltaNet, forward-Euler on the recall gradient
# flow), Longhorn's implicit step via its self-limiting effective rate, the
# closed-form k-direction spectral radii, and the exact geometric error decay
# under a repeated pair. Mirrors companions/ch12/jax/{delta_rule,longhorn,
# stability}.py; the load-bearing numbers (rho = |1 - beta*||k||^2|, the
# boundary at 2, rho_LH = alpha/(alpha + ||k||^2) < 1) must agree with the JAX
# companion to machine precision.
#
# Port credit: formulas from post_transformers/experiments/jax/week12/
# stability_analysis.py (arXiv:2406.06484, arXiv:2407.14207).

using LinearAlgebra

"""
    delta_rule_step(S, k, v, beta)

One delta-rule update `S + beta * (v - S*k) * k'` (rank-one form): a single
explicit gradient step on the recall objective `0.5 * ||S*k - v||^2`.
"""
function delta_rule_step(S::AbstractMatrix, k::AbstractVector, v::AbstractVector, beta::Real)
    size(S) == (length(v), length(k)) ||
        throw(ArgumentError("S must be (d_v, d_k) = ($(length(v)), $(length(k))); got $(size(S))"))
    return S + beta * (v - S * k) * k'
end

"""
    delta_rule_fixed_point(k, v)

The unique fixed point `S* = v * k' / ||k||^2` (exact retrieval `S*k = v`).
"""
function delta_rule_fixed_point(k::AbstractVector, v::AbstractVector)
    ksq = dot(k, k)
    ksq > 0 || throw(ArgumentError("key must be nonzero: the fixed point is undefined"))
    return v * k' / ksq
end

"""
    longhorn_effective_beta(k, alpha)

Longhorn's self-limiting rate `1 / (alpha + ||k||^2)`, capped at `1/alpha`.
"""
function longhorn_effective_beta(k::AbstractVector, alpha::Real)
    alpha > 0 || throw(ArgumentError("alpha must be strictly positive; got $alpha"))
    return 1.0 / (alpha + dot(k, k))
end

"""
    longhorn_step(S, k, v, alpha)

One implicit-step update: the delta rule evaluated at the effective rate.
"""
function longhorn_step(S::AbstractMatrix, k::AbstractVector, v::AbstractVector, alpha::Real)
    return delta_rule_step(S, k, v, longhorn_effective_beta(k, alpha))
end

"""
    deltanet_spectral_radius(beta, ksq)

DeltaNet's k-direction spectral radius `|1 - beta * ||k||^2|`; stability
requires `beta * ||k||^2 in (0, 2)`.
"""
deltanet_spectral_radius(beta::Real, ksq::Real) = abs(1.0 - beta * ksq)

"""
    longhorn_spectral_radius(alpha, ksq)

Longhorn's k-direction spectral radius `alpha / (alpha + ||k||^2) < 1` for
every `alpha > 0` and key magnitude — unconditional stability.
"""
longhorn_spectral_radius(alpha::Real, ksq::Real) = alpha / (alpha + ksq)

"""
    a_stability_boundary()

The explicit-step boundary `beta * ||k||^2 = 2` as a named constant.
"""
a_stability_boundary() = 2.0

"""
    iteration_eigenvalue_along_k(k, beta_eff)

Rayleigh quotient of the materialised iteration matrix `I - beta_eff * k * k'`
in the `k` direction: equals `1 - beta_eff * ||k||^2` (signed). The
derivation-drift guard.
"""
function iteration_eigenvalue_along_k(k::AbstractVector, beta_eff::Real)
    ksq = dot(k, k)
    ksq > 0 || throw(ArgumentError("key must be nonzero for the Rayleigh quotient"))
    M = I - beta_eff * (k * k')
    return dot(k, M * k) / ksq
end

"""
    error_trajectory(k, v, nsteps; beta = nothing, alpha = nothing)

Frobenius deviation `||S_t - S*||` under the repeated pair `(k, v)` from
`S_0 = 0`, for exactly one of the explicit rate `beta` (DeltaNet) or the
trust-region weight `alpha` (Longhorn). The update is affine in `S`, so the
deviation is an exact geometric sequence with ratio
`|1 - beta_eff * ||k||^2|` — the spectral radius, measured.
"""
function error_trajectory(k::AbstractVector, v::AbstractVector, nsteps::Integer;
                          beta = nothing, alpha = nothing)
    ((beta === nothing) == (alpha === nothing)) &&
        throw(ArgumentError("give exactly one of beta (DeltaNet) or alpha (Longhorn)"))
    Sstar = delta_rule_fixed_point(k, v)
    S = zeros(eltype(Sstar), size(Sstar))
    norms = Vector{Float64}(undef, nsteps + 1)
    norms[1] = norm(S - Sstar)
    for t in 1:nsteps
        S = beta === nothing ? longhorn_step(S, k, v, alpha) : delta_rule_step(S, k, v, beta)
        norms[t + 1] = norm(S - Sstar)
    end
    return norms
end
