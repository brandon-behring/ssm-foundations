# lyapunov_crosscheck.jl --- Chapter 15 Julia companion (stdlib only).
#
# An independent-language implementation of the §15.4 Lyapunov diagnostic: the
# Benettin QR algorithm and the closed-form spectrum, mirroring
# companions/ch15/jax/lyapunov_diagnostics.py. Lyapunov spectra are a canonical
# numerical-analysis object, so a second, independent implementation (Julia's
# LinearAlgebra.qr / eigvals vs JAX's) is a genuine cross-check of the diagnostic,
# not ceremony. The load-bearing facts must agree with the JAX companion:
#   * a diagonal transition's spectrum is recovered exactly (QR is trivial) and
#     equals the JAX-pinned values to < 1e-9 (the crisp cross-language anchor);
#   * a Chapter 13 DPLR transition's closed-form spectrum (eigvals) matches JAX to
#     < 1e-9, and the QR estimate matches that closed form to the O(1/T) tolerance;
#   * the divergence identity sum(lambda) = log|det J| holds (autonomous case);
#   * the effective state size of the two-level spectrum [1]^r || [w]^(d-r) equals
#     the closed form (r + (d-r)w^2)^2 / (r + (d-r)w^4) (P3'), == JAX.
#
# Port credit: greenfield; the engine mirrors Chapter 2's qr_lyapunov (Benettin et
# al. 1980) and the constructed systems mirror the JAX module's literals.

using LinearAlgebra

"""
    qr_lyapunov(jacobians, n_steps)

Benettin QR algorithm: Lyapunov spectrum (descending) of a sequence of per-step
Jacobians `jacobians::Vector` of `N×N` matrices, cycled if shorter than `n_steps`.
Re-orthonormalizes the propagated frame each step (`Q_{t+1} R_t = J_t Q_t`) and
accumulates the log-stretch `sum_t log|diag(R_t)|`. Mirrors the JAX/Chapter 2 core.
"""
function qr_lyapunov(jacobians::Vector{<:AbstractMatrix}, n_steps::Int)
    n_steps >= 1 || throw(ArgumentError("n_steps must be >= 1"))
    T = length(jacobians)
    N = size(jacobians[1], 1)
    Q = Matrix{Float64}(I, N, N)
    acc = zeros(Float64, N)
    for t in 1:n_steps
        Jt = jacobians[((t - 1) % T) + 1]
        F = qr(Jt * Q)
        Qn = Matrix(F.Q)
        R = F.R
        s = sign.(diag(R))
        s[s .== 0] .= 1.0
        Qn = Qn .* transpose(s)      # sign-fix columns of Q
        R = s .* R                   # sign-fix rows of R
        acc .+= log.(abs.(diag(R)) .+ 1e-300)
        Q = Qn
    end
    return sort(acc ./ n_steps; rev = true)
end

"""
    closed_form_log_growth(J)

Closed-form Lyapunov spectrum of an autonomous discrete system: `log|λ_i(J)|`
sorted descending. Independent of the QR iteration (an eigendecomposition), the
ground-truth reference the estimator is validated against.
"""
closed_form_log_growth(J::AbstractMatrix) = sort(log.(abs.(eigvals(Matrix(J)))); rev = true)

"""
    dplr_transition(w, a, c)

The Chapter 13 diagonal-plus-rank-one transition `Diag(w) - c * a * a'` (symmetric).
"""
dplr_transition(w::AbstractVector, a::AbstractVector, c::Real) =
    Matrix(Diagonal(w)) .- c .* (a * transpose(a))

"""
    effective_state_size(magnitudes)

Participation ratio of the squared spectral magnitudes `(Σ p)^2 / Σ p^2`,
`p_i = |λ_i|^2` — a soft count of the dominant modes (P3').
"""
function effective_state_size(magnitudes::AbstractVector)
    p = abs.(magnitudes) .^ 2
    denom = sum(p .^ 2)
    denom == 0 && throw(ArgumentError("all magnitudes are zero; D_eff undefined"))
    return sum(p)^2 / denom
end

"""
    effective_state_size_closed_form(r, d, w)

`(r + (d-r) w^2)^2 / (r + (d-r) w^4)` — the two-level closed form, → r as w→0.
"""
effective_state_size_closed_form(r::Integer, d::Integer, w::Real) =
    (r + (d - r) * w^2)^2 / (r + (d - r) * w^4)
