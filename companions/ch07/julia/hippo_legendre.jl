# hippo_legendre.jl --- Chapter 7 Julia companion: HiPPO-LegS projection operator.
#
# Fresh implementation (no direct Julia source) in the style of
# companions/ch04/julia/discretization_atlas.jl: stdlib `LinearAlgebra` only, so it
# joins the quick ch05/ch06 test loop with no heavy precompilation.
#
# Mirrors the JAX companions (companions/ch07/jax/hippo_matrix.py,
# hippo_reconstruction.py). Cross-language contrast: where the JAX encoder threads the
# coefficient vector through `jax.lax.scan` (one fused pass), the idiomatic Julia
# spelling is an explicit `for` loop — Julia's JIT compiles the loop body to native
# code, so the loop *is* the performant program (no scan primitive needed).
#
# Closed form (normalized scaled-Legendre, math indices p,q starting at 0):
#   A[p,q] = -sqrt((2p+1)(2q+1))  for p>q ;  -(p+1) for p==q ;  0 for p<q
#   B[p]   =  sqrt(2p+1)
# A is lower-triangular, so eigenvalues are exactly -1,-2,...,-N (stable).
#
# Usage:
#     julia --project=companions/ch07/julia companions/ch07/julia/hippo_legendre.jl

using LinearAlgebra

"""
    make_hippo_legs(n) -> (A, B)

HiPPO-LegS matrices: `A` is `n×n` lower-triangular (eigenvalues -1..-N), `B` is `n×1`.
Throws `ArgumentError` if `n < 1`.
"""
function make_hippo_legs(n::Int)
    n >= 1 || throw(ArgumentError("state dimension n must be >= 1, got $n"))
    A = zeros(Float64, n, n)
    for i in 1:n, j in 1:n
        p, q = i - 1, j - 1          # 0-based math indices
        if p > q
            A[i, j] = -sqrt((2p + 1) * (2q + 1))
        elseif p == q
            A[i, j] = -(p + 1)
        end
    end
    B = reshape([sqrt(2 * (i - 1) + 1) for i in 1:n], n, 1)
    return A, B
end

"""
    legs_eigenvalues(n) -> Vector

Eigenvalues of the HiPPO-LegS `A` (real; equal to the diagonal -1,...,-N).
"""
legs_eigenvalues(n::Int) = eigvals(make_hippo_legs(n)[1])

"""
    legendre_basis(n, z; normalized=true) -> Matrix (n × length(z))

Shifted Legendre basis, row `r` = degree `r-1`, evaluated at `2z-1` via the stable
3-term recurrence. `normalized=true` scales row of degree `d` by `sqrt(2d+1)`.
"""
function legendre_basis(n::Int, z::AbstractVector{<:Real}; normalized::Bool = true)
    x = 2 .* z .- 1
    P = zeros(Float64, n, length(z))
    P[1, :] .= 1.0
    if n >= 2
        P[2, :] .= x
    end
    for r in 3:n                      # degree D = r-1
        @views P[r, :] .= ((2r - 3) .* x .* P[r - 1, :] .- (r - 2) .* P[r - 2, :]) ./ (r - 1)
    end
    if normalized
        for r in 1:n
            P[r, :] .*= sqrt(2 * (r - 1) + 1)
        end
    end
    return P
end

"""
    hippo_legs_encode(u, n) -> Matrix (length(u) × n)

Online LegS encoding of signal samples `u` (on a uniform grid of [0,1]) into Legendre
coefficients, via the bilinear time-varying recurrence
`(I + A_pos/2k) c_k = (I - A_pos/2k) c_{k-1} + (B/k) u_k` with `A_pos = -A`. Returns the
full coefficient trajectory (final row reconstructs the whole history). Explicit loop —
the cross-language contrast to the JAX `lax.scan`.
"""
function hippo_legs_encode(u::AbstractVector{<:Real}, n::Int)
    A, B = make_hippo_legs(n)
    Apos = -A
    Bv = vec(B)
    eyeN = Matrix{Float64}(I, n, n)
    L = length(u)
    traj = zeros(Float64, L, n)
    c = zeros(Float64, n)
    for k in 1:L
        lhs = eyeN + Apos / (2k)
        rhs = (eyeN - Apos / (2k)) * c + (Bv / k) * u[k]
        c = lhs \ rhs
        traj[k, :] = c
    end
    return traj
end

"""
    reconstruct(c, z) -> Vector

Reconstruct the history `û(z) = Σ_n c_n sqrt(2n+1) P_n(2z-1)` (normalized,
non-alternating — the converging LegS convention calibrated in the JAX companion).
"""
reconstruct(c::AbstractVector{<:Real}, z::AbstractVector{<:Real}) =
    vec(c' * legendre_basis(length(c), z; normalized = true))

"Smooth band-limited test history: two sinusoids on z ∈ [0,1]."
test_signal(z::AbstractVector{<:Real}) = sin.(2π .* 1.5 .* z) .+ 0.5 .* sin.(2π .* 4.0 .* z)

"Relative L2 reconstruction error at the final time for state dimension n."
function reconstruction_error(n::Int; L::Int = 1000)
    z = range(0.0, 1.0; length = L) |> collect
    truth = test_signal(z)
    c_final = hippo_legs_encode(truth, n)[end, :]
    approx = reconstruct(c_final, z)
    return norm(approx - truth) / norm(truth)
end

# --- direct-run demo (skipped when included as a module by runtests.jl) ---
if abspath(PROGRAM_FILE) == @__FILE__
    println("Chapter 7 — hippo_legendre.jl")
    println("="^60)
    for n in (4, 8, 16)
        eigs = sort(real(legs_eigenvalues(n)))
        println("  N=$n eigenvalues (sorted real) = ", round.(eigs; digits = 3))
    end
    println("  Reconstruction error vs N:")
    for n in (4, 8, 16, 32, 64)
        println("    N=$n: ", round(reconstruction_error(n); sigdigits = 4))
    end
end
