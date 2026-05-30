# discretization.jl
#
# Chapter 10 companion (Julia): order of accuracy and stability for the three
# scalar discretizations behind Mamba-3's integrator choice — zero-order hold
# (ZOH), bilinear (Tustin), and exponential-trapezoidal (§4.5, the Mamba-3
# scheme). The Julia sibling of `companions/ch10/jax/discretization.py`.
#
# Scope: the numerical-analysis CORE only (the C1 symplectic-integrator pilot's
# language). The SSD / complex-state pieces stay in the JAX + torch companions;
# this module is the discretization atlas that joins the Ch 4-6 Julia family.
#
# STDLIB-ONLY by design (LinearAlgebra/Printf/Test): unlike Chapter 4's
# `discretization_atlas.jl`, which benchmarks against a DifferentialEquations.jl
# Tsit5 solve, this module uses the *analytic* variation-of-constants solution of
# the scalar forced problem  x' = A x + sin(ωt)  as ground truth. That keeps it
# in the fast `make companion-julia-tests` loop (which excludes ch04 precisely
# because of its heavy DifferentialEquations dependency).
#
# THE load-bearing subtlety (§10.2), made visible here exactly as in the JAX
# companion: for exponential integrators the transition α = exp(A·dt) is exact,
# so on a HOMOGENEOUS system (u ≡ 0) ZOH and exp-trapezoidal are *identical* and
# the order difference is INVISIBLE. The order-2 gain lives entirely in the input
# quadrature, so it is measurable only on a FORCED system. `order_sweep` measures
# the forced case; `homogeneous_error` demonstrates the blindness.
#
# Port credit: structure mirrors `companions/ch04/julia/discretization_atlas.jl`
# and the JAX `companions/ch10/jax/discretization.py`. Mamba-3: Lahoti et al.,
# arXiv:2603.15569. Exponential integrators: Hochbruck & Ostermann, Acta
# Numerica 2010.
#
# Usage:
#     julia --project=companions/ch10/julia companions/ch10/julia/discretization.jl

using LinearAlgebra
using Printf

# ---------------------------------------------------------------------------
# Scalar discretizations of  dx/dt = A x + u(t).
#
# Each returns the coefficients of its one-step update. A may be complex (a
# single Mamba-3 eigenvalue ρ·e^{iθ}); dt is real and positive.
# ---------------------------------------------------------------------------

"""
    discretize_zoh(A, dt) -> (α, β)

Zero-order hold (first-order). `x_k = α x_{k-1} + β u_{k-1}`, with α = exp(A dt)
exact and β = (α - 1)/A (→ dt as A → 0). Exact on the homogeneous part.
"""
function discretize_zoh(A::Number, dt::Real)
    α = exp(A * dt)
    β = abs(A) < 1e-12 ? complex(dt) : (α - 1) / A
    return (α, β)
end

"""
    discretize_bilinear(A, dt) -> (α, β)

Bilinear / Tustin (second-order, A-stable). `x_k = α x_{k-1} + β (u_{k-1}+u_k)`
with α = (1 + Adt/2)/(1 - Adt/2) the (1,1)-Padé approximation of exp(A dt) and
β = (dt/2)/(1 - Adt/2) on each endpoint. α only approximates the exponential, so
on stiff modes α → -1 (undamped), unlike the exponential schemes.
"""
function discretize_bilinear(A::Number, dt::Real)
    half = dt / 2
    denom = 1 - half * A
    α = (1 + half * A) / denom
    β = half / denom
    return (α, β)
end

"""
    discretize_exp_trapezoidal(A, dt; λ=0.5) -> (α, β, γ)

Exponential-trapezoidal (second-order), Mamba-3's integrator. The three-tuple
update `x_k = α x_{k-1} + β u_{k-1} + γ u_k` with α = exp(A dt) exact,
β = (1-λ) dt α (left endpoint, decayed across the step), γ = λ dt (right
endpoint). λ = 1/2 is the symmetric trapezoid (order 2); λ = 1 degenerates to a
shifted first-order ZOH.
"""
function discretize_exp_trapezoidal(A::Number, dt::Real; λ::Real = 0.5)
    0 <= λ <= 1 || throw(ArgumentError("λ must be in [0, 1], got $λ"))
    α = exp(A * dt)
    β = (1 - λ) * dt * α
    γ = λ * dt
    return (α, β, γ)
end

# ---------------------------------------------------------------------------
# Forced exact solution + integration + order measurement.
# ---------------------------------------------------------------------------

"""
    forced_exact(A, ω, t; x0=0) -> x(t)

Exact solution of  x' = A x + sin(ω t),  x(0) = x0, by variation of constants:

    x(t) = (x0 + ω/(A²+ω²)) e^{At} - (A sin ωt + ω cos ωt)/(A²+ω²).

Used as the stdlib-only ground truth for the order sweep.
"""
function forced_exact(A::Number, ω::Real, t::Real; x0::Number = 0)
    denom = A^2 + ω^2
    abs(denom) > 1e-300 || throw(ArgumentError("A² + ω² must be nonzero"))
    c = x0 + ω / denom
    return c * exp(A * t) - (A * sin(ω * t) + ω * cos(ω * t)) / denom
end

"""
    integrate(scheme, A, dt, n_steps, ω; x0=0, λ=0.5) -> Vector

Integrate  x' = A x + u(t)  for `n_steps` steps of size `dt`. `u(t) = sin(ω t)`
when `ω` is a real number, or `u ≡ 0` when `ω === nothing` (the homogeneous case
that exposes the order-blindness). Returns states x_0 … x_{n_steps}.
"""
function integrate(scheme::Symbol, A::Number, dt::Real, n_steps::Integer,
                   ω::Union{Real,Nothing}; x0::Number = 0, λ::Real = 0.5)
    T = promote_type(typeof(complex(A)), ComplexF64)
    xs = Vector{T}(undef, n_steps + 1)
    xs[1] = x0
    u(k) = ω === nothing ? 0.0 : sin(ω * (k * dt))  # u at time k*dt

    if scheme === :zoh
        α, β = discretize_zoh(A, dt)
        for k in 1:n_steps
            xs[k+1] = α * xs[k] + β * u(k - 1)
        end
    elseif scheme === :bilinear
        α, β = discretize_bilinear(A, dt)
        for k in 1:n_steps
            xs[k+1] = α * xs[k] + β * (u(k - 1) + u(k))
        end
    elseif scheme === :exp_trapezoidal
        α, β, γ = discretize_exp_trapezoidal(A, dt; λ = λ)
        for k in 1:n_steps
            xs[k+1] = α * xs[k] + β * u(k - 1) + γ * u(k)
        end
    else
        throw(ArgumentError("unknown scheme $scheme"))
    end
    return xs
end

"""
    global_error(scheme, A, dt, T_end, ω; x0=0, λ=0.5) -> Float64

Max absolute error over [0, T_end] between a scheme and the exact solution
(`forced_exact` for forced; x0·e^{At} for homogeneous).
"""
function global_error(scheme::Symbol, A::Number, dt::Real, T_end::Real,
                      ω::Union{Real,Nothing}; x0::Number = 0, λ::Real = 0.5)
    n_steps = Int(round(T_end / dt))
    xs = integrate(scheme, A, dt, n_steps, ω; x0 = x0, λ = λ)
    err = 0.0
    for k in 0:n_steps
        t = k * dt
        exact = ω === nothing ? x0 * exp(A * t) : forced_exact(A, ω, t; x0 = x0)
        err = max(err, abs(xs[k+1] - exact))
    end
    return err
end

"""
    order_sweep(scheme, A, dts, T_end, ω; x0=0, λ=0.5) -> (errors, slope)

Global error at each step size plus the empirical convergence order (slope of
log-error vs log-dt over the two finest step sizes). ≈1 for ZOH, ≈2 for
exp-trapezoidal / bilinear on the forced system.
"""
function order_sweep(scheme::Symbol, A::Number, dts::AbstractVector, T_end::Real,
                     ω::Union{Real,Nothing}; x0::Number = 0, λ::Real = 0.5)
    errors = [global_error(scheme, A, dt, T_end, ω; x0 = x0, λ = λ) for dt in dts]
    slope = log(errors[end-1] / errors[end]) / log(dts[end-1] / dts[end])
    return (errors, slope)
end

"""
    homogeneous_error(scheme, A, dt, T_end; x0=1) -> Float64

Max error of `scheme` on the autonomous system (u ≡ 0). For the exponential
schemes (ZOH, exp-trapezoidal) this is at roundoff regardless of order — the
§10.2 homogeneous-blindness.
"""
function homogeneous_error(scheme::Symbol, A::Number, dt::Real, T_end::Real; x0::Number = 1)
    return global_error(scheme, A, dt, T_end, nothing; x0 = x0)
end

"""
    amplification(scheme, z) -> Number

Amplification factor α(z) of the homogeneous recurrence as a function of
z = A·dt. exp(z) for ZOH / exp-trapezoidal (→ 0 on stiff modes), the Padé form
for bilinear (→ -1), and 1 + z for forward Euler (not A-stable). Independent of
λ: the exp-trapezoidal transition is e^z for any interpolation weight.
"""
function amplification(scheme::Symbol, z::Number)
    if scheme === :zoh || scheme === :exp_trapezoidal
        return exp(z)
    elseif scheme === :bilinear
        return (1 + z / 2) / (1 - z / 2)
    elseif scheme === :forward_euler
        return 1 + z
    else
        throw(ArgumentError("unknown scheme $scheme"))
    end
end

# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if abspath(PROGRAM_FILE) == @__FILE__
    println("Chapter 10 Julia companion: discretization order + stability")
    println("=" ^ 66)
    # One complex decaying-oscillating mode (a single Mamba-3 eigenvalue).
    A = -0.5 + 2.0im
    ω = 1.3
    T_end = 6.0
    x0 = 1.0 + 0.0im
    dts = [0.2, 0.1, 0.05, 0.025, 0.0125]

    println("System: x' = A x + sin(ωt),  A = $A,  ω = $ω")
    println()
    println("HOMOGENEOUS (u ≡ 0): exponential transition is exact for both")
    @printf("  ZOH      max error = %.2e  (roundoff)\n", homogeneous_error(:zoh, A, 0.1, T_end; x0 = x0))
    @printf("  exp-trap max error = %.2e  (roundoff)\n", homogeneous_error(:exp_trapezoidal, A, 0.1, T_end; x0 = x0))
    println("  -> order is INVISIBLE here (§10.2 homogeneous-blindness)")
    println()
    println("FORCED (u = sin ωt): order is visible")
    for (name, sym) in (("ZOH", :zoh), ("exp-trapezoidal", :exp_trapezoidal), ("bilinear", :bilinear))
        _, slope = order_sweep(sym, A, dts, T_end, ω; x0 = x0)
        @printf("  %-16s slope ≈ %.3f\n", name, slope)
    end
    println()
    z = -30.0
    @printf("STIFF mode z = %.0f:  |α| exp-trap = %.2e (→0),  |α| bilinear = %.4f (→1)\n",
            z, abs(amplification(:exp_trapezoidal, z)), abs(amplification(:bilinear, z)))
end
