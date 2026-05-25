# symplectic_methods.jl
#
# Chapter 6 Julia companion: symplectic Euler, Störmer-Verlet, and the
# 2-stage Gauss-Legendre IRK method (= implicit midpoint, order 2) on a
# Hamiltonian test problem.
#
# Test problem: harmonic oscillator with H(q, p) = (q² + p²)/2, so
#     dq/dt =  p =  ∂H/∂p,
#     dp/dt = -q = -∂H/∂q.
# Exact: q(t) = q_0 cos t + p_0 sin t, p(t) = -q_0 sin t + p_0 cos t.
# Energy is exactly preserved by the continuous flow.
#
# What this companion measures:
#   1. Empirical order of accuracy on the harmonic-oscillator state at t = 2π.
#   2. Long-horizon energy drift: each method run for 100 periods, max
#      |E(t) - E_0| reported.
#
# Symplectic methods preserve a modified Hamiltonian and exhibit *bounded*
# energy oscillation (no secular drift). Standard RK methods of the same
# order have linear-in-time drift. The empirical numbers below make this
# concrete.
#
# Usage:
#     julia --project=. companions/ch06/julia/symplectic_methods.jl

using LinearAlgebra
using Printf

# ---------------------------------------------------------------------------
# Hamiltonian system: harmonic oscillator
# ---------------------------------------------------------------------------

H_energy(q::Float64, p::Float64) = 0.5 * (q * q + p * p)
T_grad(p::Float64) = p          # ∂T/∂p where T(p) = p²/2
V_grad(q::Float64) = q          # ∂V/∂q where V(q) = q²/2

# ---------------------------------------------------------------------------
# Symplectic Euler — order 1, "p first" variant.
#   p_{n+1} = p_n - Δ · V'(q_n)
#   q_{n+1} = q_n + Δ · T'(p_{n+1})
# ---------------------------------------------------------------------------

function symplectic_euler_step(q::Float64, p::Float64, dt::Float64)
    p_next = p - dt * V_grad(q)
    q_next = q + dt * T_grad(p_next)
    return q_next, p_next
end

# ---------------------------------------------------------------------------
# Störmer-Verlet — order 2, symmetric, time-reversible.
#   p_{n+1/2} = p_n - (Δ/2) V'(q_n)
#   q_{n+1}   = q_n + Δ T'(p_{n+1/2})
#   p_{n+1}   = p_{n+1/2} - (Δ/2) V'(q_{n+1})
# ---------------------------------------------------------------------------

function verlet_step(q::Float64, p::Float64, dt::Float64)
    p_half = p - 0.5 * dt * V_grad(q)
    q_next = q + dt * T_grad(p_half)
    p_next = p_half - 0.5 * dt * V_grad(q_next)
    return q_next, p_next
end

# ---------------------------------------------------------------------------
# 2-stage Gauss-Legendre IRK == implicit midpoint rule, order 2, A-stable + symplectic.
#
# For the linear harmonic-oscillator system, the implicit midpoint rule
# reduces to
#     x_{n+1} = (I - (dt/2) M)⁻¹ (I + (dt/2) M) x_n,
# where x = [q, p] and M = [[0, 1], [-1, 0]]. This is the bilinear (Tustin)
# transform — which is *also* symplectic for this system (a happy coincidence
# specific to the harmonic oscillator; bilinear is not symplectic in general).
# ---------------------------------------------------------------------------

const M_HO = Float64[0.0  1.0; -1.0  0.0]

function gauss_legendre_step(q::Float64, p::Float64, dt::Float64)
    Id = Matrix{Float64}(I, 2, 2)
    half = 0.5 * dt
    L = Id - half .* M_HO
    R = Id + half .* M_HO
    update = L \ (R * [q, p])
    return update[1], update[2]
end

# ---------------------------------------------------------------------------
# Classical RK4 — non-symplectic, order 4 (reference)
# ---------------------------------------------------------------------------

function rk4_step(q::Float64, p::Float64, dt::Float64)
    f(state) = (T_grad(state[2]), -V_grad(state[1]))
    k1q, k1p = f((q, p))
    k2q, k2p = f((q + 0.5 * dt * k1q, p + 0.5 * dt * k1p))
    k3q, k3p = f((q + 0.5 * dt * k2q, p + 0.5 * dt * k2p))
    k4q, k4p = f((q + dt * k3q, p + dt * k3p))
    q_next = q + (dt / 6.0) * (k1q + 2 * k2q + 2 * k3q + k4q)
    p_next = p + (dt / 6.0) * (k1p + 2 * k2p + 2 * k3p + k4p)
    return q_next, p_next
end

# ---------------------------------------------------------------------------
# Simulation harness
# ---------------------------------------------------------------------------

function simulate(step_fn, q0::Float64, p0::Float64, dt::Float64, n_steps::Int)
    q, p = q0, p0
    energies = zeros(Float64, n_steps + 1)
    energies[1] = H_energy(q, p)
    for k in 1:n_steps
        q, p = step_fn(q, p, dt)
        energies[k + 1] = H_energy(q, p)
    end
    return q, p, energies
end

# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

function order_table()
    println("=" ^ 70)
    println("Order of accuracy on harmonic oscillator at t = 1.5 (mid-orbit)")
    println("=" ^ 70)
    # Mid-orbit endpoint (not at multiple of 2π) so phase error doesn't
    # contaminate the leading-order amplitude error.
    dts = Float64[0.1, 0.05, 0.025, 0.0125, 0.00625]
    t_end = 1.5
    q0, p0 = 1.0, 0.0
    q_exact = cos(t_end) * q0 + sin(t_end) * p0
    p_exact = -sin(t_end) * q0 + cos(t_end) * p0

    methods = [
        ("Symplectic Euler", symplectic_euler_step, 1),
        ("Verlet",           verlet_step,            2),
        ("Gauss-Legendre 2", gauss_legendre_step,   2),
        ("Classical RK4",    rk4_step,               4),
    ]

    for (name, step_fn, expected_order) in methods
        println("\n$name (expected order $expected_order)")
        println("-" ^ 40)
        errs = Float64[]
        for dt in dts
            n_steps = Int(round(t_end / dt))
            q, p, _ = simulate(step_fn, q0, p0, dt, n_steps)
            err = max(abs(q - q_exact), abs(p - p_exact))
            push!(errs, err)
            @printf("  dt = %5.3f   err = %.4e\n", dt, err)
        end
        slope = log(errs[end-1] / errs[end]) / log(dts[end-1] / dts[end])
        @printf("  empirical slope ≈ %.3f\n", slope)
    end
end

function energy_drift_table()
    println()
    println("=" ^ 70)
    println("Long-horizon energy drift on harmonic oscillator (100 periods)")
    println("=" ^ 70)
    dt = 0.05
    periods = 100
    n_steps = Int(round(periods * 2π / dt))
    q0, p0 = 1.0, 0.0
    E0 = H_energy(q0, p0)

    methods = [
        ("Symplectic Euler", symplectic_euler_step),
        ("Verlet",           verlet_step),
        ("Gauss-Legendre 2", gauss_legendre_step),
        ("Classical RK4",    rk4_step),
    ]

    @printf("%-22s  %-15s  %-15s  %-15s\n", "Method", "E_max-E_0", "E_min-E_0", "E_final-E_0")
    println("-" ^ 78)
    for (name, step_fn) in methods
        _, _, energies = simulate(step_fn, q0, p0, dt, n_steps)
        E_max = maximum(energies)
        E_min = minimum(energies)
        E_final = energies[end]
        @printf("%-22s  %+.4e   %+.4e   %+.4e\n", name, E_max - E0, E_min - E0, E_final - E0)
    end
    println()
    println("Interpretation:")
    println("  Symplectic Euler / Verlet / Gauss-Legendre: E_max - E_min bounded")
    println("    by O(dt^p) (1, 2, 2 respectively); E_final stays close to E_0.")
    println("  RK4: E_final drifts linearly — much larger than its O(dt^4) local")
    println("    error would suggest, because of secular accumulation.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    println("Chapter 6 Julia companion: symplectic vs non-symplectic integrators")
    println("=" ^ 70)
    order_table()
    energy_drift_table()
end
