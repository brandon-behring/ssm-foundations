# discretization_atlas.jl
#
# Chapter 4 companion (Julia): ZOH, bilinear (Tustin), and
# exponential-trapezoidal discretizations of the forced linear oscillator,
# benchmarked against a high-accuracy DifferentialEquations.jl Tsit5 reference.
#
# Ported and extended from `post_transformers/experiments/julia/week09/
# discretization.jl`. The book-version uses Float64 throughout (the
# post_transformers source uses Float32 to match SSM training-time precision)
# and adds a tabulated summary keyed to Table 4.1 of Chapter 4.
#
# The test system is the same forced damped oscillator as the JAX companions:
#     d/dt [q, q̇]ᵀ = [[0, 1], [-4, -0.5]] [q, q̇]ᵀ + [0, 1]ᵀ · sin(2t),
#     y = [1, 0] · [q, q̇]ᵀ.
# Eigenvalues of A: -0.25 ± i·√(15)/4 (firmly in the open left half-plane).
#
# Usage:
#     julia --project=. companions/ch04/julia/discretization_atlas.jl

using DifferentialEquations
using LinearAlgebra
using Printf

# ---------------------------------------------------------------------------
# Test system
# ---------------------------------------------------------------------------

const A_OSC = Float64[0.0  1.0; -4.0 -0.5]
const B_OSC = Float64[0.0, 1.0]
const C_OSC = Float64[1.0, 0.0]

"Smooth scalar input u(t) = sin(2t) — same forcing as Chapter 4 §4.3."
drive(t) = sin(2.0 * t)

# ---------------------------------------------------------------------------
# Discretizers
#
# All three return a NamedTuple `(Ad, ...)` and pair with a `step_*` function;
# the harness in `simulate` calls them uniformly.
# ---------------------------------------------------------------------------

"""
    discretize_zoh(A, B, dt) -> (Ad, Bd)

Zero-order hold via the augmented matrix exponential trick:

    exp(dt · [A B; 0 0]) = [Ad  Bd; 0  I].

Avoids inverting A. Accuracy: first-order on forced systems, exact on
autonomous (u ≡ 0). A-stable.
"""
function discretize_zoh(A::AbstractMatrix{T}, B::AbstractVector{T}, dt::T) where {T}
    n = size(A, 1)
    M = zeros(T, n + 1, n + 1)
    M[1:n, 1:n] .= A
    M[1:n, n+1] .= B
    E = exp(M .* dt)
    return (Ad = E[1:n, 1:n], Bd = E[1:n, n+1])
end

"""
    discretize_bilinear(A, B, dt) -> (Ad, Bd)

Tustin transform — second-order accurate, A-stable, maps the imaginary axis
exactly to the unit circle. Uses the input midpoint `(u_k + u_{k+1})/2` to
realize true second-order convergence (using `u_k` alone degrades to first
order; this was confirmed empirically while debugging the post_transformers
Week 9 source).
"""
function discretize_bilinear(A::AbstractMatrix{T}, B::AbstractVector{T}, dt::T) where {T}
    n = size(A, 1)
    Id = Matrix{T}(I, n, n)
    half = T(0.5) * dt
    L = Id - half .* A
    Ad = L \ (Id + half .* A)
    Bd = L \ (dt .* B)
    return (Ad = Ad, Bd = Bd)
end

"""
    discretize_exp_trap(A, B, dt) -> (Ad, B0, B1)

Exponential-trapezoidal scheme of §4.5: exact matrix exponential on the
homogeneous part + trapezoidal-rule linear interpolation of the forcing.

    h_{k+1} = exp(A dt) h_k
            + dt · φ₁(A dt) B · u_k
            + dt · φ₂(A dt) B · (u_{k+1} - u_k),

with φ₁(z) = (eᶻ - 1)/z and φ₂(z) = (eᶻ - 1 - z)/z². Computed via the
augmented-matrix-exponential trick (Al-Mohy & Higham 2011) for numerical
stability at small dt. Accuracy: second-order for C¹ inputs. A-stable.
"""
function discretize_exp_trap(A::AbstractMatrix{T}, B::AbstractVector{T}, dt::T) where {T}
    n = size(A, 1)
    # Construct M = dt · Â where Â = [[A, B, 0]; [0 0 1]; [0 0 0]], so
    # M = [[A·dt, B·dt, 0]; [0 0 dt]; [0 0 0]] (shape (n+2, n+2)). The
    # (n+1, n+2) entry MUST be `dt`, not `1`: see the Python companion
    # `exp_trapezoidal.py:105-109` for the derivation. The previous
    # `T(1.0)` here was a bug (F29 in audits/2026-05-27_repo_audit_deeper.md)
    # that silently degraded exp-trap from second- to first-order accuracy.
    M = zeros(T, n + 2, n + 2)
    M[1:n, 1:n] .= A .* dt
    M[1:n, n+1] .= B .* dt
    M[n+1, n+2] = dt
    E = exp(M)
    Ad = E[1:n, 1:n]
    B0 = E[1:n, n+1]           # dt · φ₁(A dt) · B
    B1_scaled = E[1:n, n+2]    # dt² · φ₂(A dt) · B
    B1 = B1_scaled ./ dt       # dt · φ₂(A dt) · B
    return (Ad = Ad, B0 = B0, B1 = B1)
end

# ---------------------------------------------------------------------------
# Step functions
# ---------------------------------------------------------------------------

step_zoh(p, h, u_k, _)        = p.Ad * h + p.Bd * u_k
step_bilinear(p, h, u_k, u_kp1) = p.Ad * h + p.Bd * 0.5 * (u_k + u_kp1)
step_exp_trap(p, h, u_k, u_kp1) = p.Ad * h + p.B0 * u_k + p.B1 * (u_kp1 - u_k)

# ---------------------------------------------------------------------------

"""
    simulate(discretize_fn, step_fn, dt, t_end) -> (ts, ys)

Run the discrete recurrence with the chosen `discretize_fn` / `step_fn` pair
on the forced oscillator over `[0, t_end]` starting from `h = 0`.
"""
function simulate(discretize_fn, step_fn, dt::T, t_end::T) where {T}
    params = discretize_fn(A_OSC, B_OSC, dt)
    n = Int(round(t_end / dt)) + 1
    h = zeros(T, 2)
    ts = T.(dt .* (0:n-1))
    ys = zeros(T, n)
    for k in 1:n-1
        u_curr = T(drive(ts[k]))
        u_next = T(drive(ts[k+1]))
        ys[k] = dot(C_OSC, h)
        h = step_fn(params, h, u_curr, u_next)
    end
    ys[n] = dot(C_OSC, h)
    return ts, ys
end

"""
    continuous_forced(t_end; reltol, abstol) -> ODESolution

High-accuracy reference via Tsit5 (5th-order RK with adaptive step).
"""
function continuous_forced(t_end::Float64; reltol = 1e-11, abstol = 1e-13)
    function rhs!(dh, h, _p, t)
        mul!(dh, A_OSC, h)
        dh .+= B_OSC .* drive(t)
        return nothing
    end
    prob = ODEProblem(rhs!, zeros(Float64, 2), (0.0, t_end))
    return solve(prob, Tsit5(); reltol = reltol, abstol = abstol)
end

# ---------------------------------------------------------------------------

"""
    error_sweep() -> Dict{String, NamedTuple}

Run the error-vs-step-size sweep for the three Chapter-4 discretizations.
Returns a dict mapping scheme name to `(dts, errs, slope)` and prints a
table of max pointwise errors plus the empirical convergence order.

The slope is computed from the two finest step sizes (where the method's
leading-order error dominates over floating-point roundoff).
"""
function error_sweep()
    dts = Float64[0.4, 0.2, 0.1, 0.05, 0.025]
    t_end = 4.0
    ref = continuous_forced(t_end)

    methods = (
        ("ZOH",       discretize_zoh,       step_zoh),
        ("Bilinear",  discretize_bilinear,  step_bilinear),
        ("Exp-trap",  discretize_exp_trap,  step_exp_trap),
    )

    println("Max |y_discrete - y_Tsit5| on forced oscillator, u(t) = sin(2t)")
    println("=" ^ 70)
    @printf("%-10s", "dt")
    for (name, _, _) in methods
        @printf("%-16s", name)
    end
    println()
    println("-" ^ 70)

    errs = Dict(name => Float64[] for (name, _, _) in methods)

    for dt in dts
        @printf("%-10.4f", dt)
        for (name, disc_fn, step_fn) in methods
            ts, ys = simulate(disc_fn, step_fn, dt, t_end)
            yref = [dot(C_OSC, ref(t)) for t in ts]
            err = maximum(abs.(ys .- yref))
            push!(errs[name], err)
            @printf("%-16.6e", err)
        end
        println()
    end

    println()
    println("Empirical convergence order (slope from finest two dt):")
    println("-" ^ 50)
    slopes = Dict{String, Float64}()
    for (name, _, _) in methods
        e = errs[name]
        slope = log(e[end-1] / e[end]) / log(dts[end-1] / dts[end])
        slopes[name] = slope
        expected = name == "ZOH" ? "(expect ~1)" : "(expect ~2)"
        @printf("  %-12s  slope ≈ %5.3f   %s\n", name, slope, expected)
    end

    return Dict(name => (dts = dts, errs = errs[name], slope = slopes[name])
                for (name, _, _) in methods)
end

# ---------------------------------------------------------------------------

if abspath(PROGRAM_FILE) == @__FILE__
    println("Chapter 4 Julia companion: discretization atlas")
    println("=" ^ 70)
    println("System: 2-state forced damped oscillator")
    println("        d/dt [q, q̇] = [[0, 1], [-4, -0.5]] [q, q̇] + [0, 1] sin(2t)")
    println("Eigenvalues of A: -0.25 ± i·√(15)/4 ≈ -0.25 ± 0.968i")
    println("Reference: Tsit5 with reltol=1e-11, abstol=1e-13")
    println()
    results = error_sweep()
    println()
    println("Interpretation (Chapter 4 Table 4.1):")
    println("  - ZOH: first-order on forced systems; would be EXACT on")
    println("    autonomous (u ≡ 0) inputs (autonomous-exactness).")
    println("  - Bilinear: second-order, A-stable, maps imaginary axis")
    println("    exactly to unit circle.")
    println("  - Exp-trapezoidal: second-order, A-stable, autonomous-exact;")
    println("    Mamba-3's chosen scheme (Chapter 10).")
end
