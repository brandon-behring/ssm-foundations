# implicit_methods.jl
#
# Chapter 6 Julia companion: backward Euler and BDF2 on a stiff linear
# test problem, with empirical order-of-accuracy verification.
#
# Test problem: 2-state linear stiff system with eigenvalues separated by
# four orders of magnitude:
#     d/dt [x_slow, x_fast] = [[-1, 0], [0, -1000]] [x_slow, x_fast].
# Exact solution: x_slow(t) = x_slow(0) e^{-t},  x_fast(t) = x_fast(0) e^{-1000 t}.
#
# Forward Euler would need dt < 2e-3 to remain stable. Backward Euler and
# BDF2 are L-stable, so any dt > 0 is admissible — though accuracy demands
# dt small enough that the slow mode resolves.
#
# Methods:
#   * Backward Euler: 1st-order, L-stable (Ch 6 §6.2).
#   * BDF2:           2nd-order, L-stable (Ch 6 §6.2).
#
# Each method is implemented from scratch (no DifferentialEquations.jl
# dependency) to keep the companion lightweight. The chapter prose discusses
# DIRK and Gauss-Legendre IRK; those are higher-stage variants of the same
# pattern.
#
# Usage:
#     julia --project=. companions/ch06/julia/implicit_methods.jl

using LinearAlgebra
using Printf

# ---------------------------------------------------------------------------
# Test problem
# ---------------------------------------------------------------------------

const A_STIFF = Float64[-1.0  0.0;
                         0.0 -1000.0]

const X0_TEST = Float64[1.0, 1.0]

"Exact solution at time t starting from x0."
exact_solution(t::Float64, x0::AbstractVector{Float64}) = [exp(-t) * x0[1], exp(-1000.0 * t) * x0[2]]

# ---------------------------------------------------------------------------
# Backward Euler — for the linear case, the implicit equation
#     x_{k+1} = x_k + dt · A · x_{k+1}
# rearranges to
#     (I - dt · A) · x_{k+1} = x_k.
# ---------------------------------------------------------------------------

function backward_euler_step(A::AbstractMatrix{Float64}, x::AbstractVector{Float64}, dt::Float64)
    n = size(A, 1)
    L = Matrix{Float64}(I, n, n) - dt .* A
    return L \ x
end

function simulate_be(A::AbstractMatrix{Float64}, x0::AbstractVector{Float64}, dt::Float64, t_end::Float64)
    n_steps = Int(round(t_end / dt))
    x = copy(x0)
    for _ in 1:n_steps
        x = backward_euler_step(A, x, dt)
    end
    return x
end

# ---------------------------------------------------------------------------
# BDF2 — uses two previous states. The recurrence is
#     (3/2) x_{k+1} - 2 x_k + (1/2) x_{k-1} = dt · A · x_{k+1},
# rearranged to
#     ((3/2) I - dt A) · x_{k+1} = 2 x_k - (1/2) x_{k-1}.
# The first step is taken with backward Euler to bootstrap.
# ---------------------------------------------------------------------------

function simulate_bdf2(A::AbstractMatrix{Float64}, x0::AbstractVector{Float64}, dt::Float64, t_end::Float64)
    n_steps = Int(round(t_end / dt))
    n = size(A, 1)
    Id = Matrix{Float64}(I, n, n)
    # Bootstrap: one backward Euler step.
    x_prev = copy(x0)
    x_curr = backward_euler_step(A, x_prev, dt)
    L = (1.5) .* Id - dt .* A
    for _ in 2:n_steps
        rhs = 2.0 .* x_curr .- 0.5 .* x_prev
        x_next = L \ rhs
        x_prev = x_curr
        x_curr = x_next
    end
    return x_curr
end

# ---------------------------------------------------------------------------
# Order-of-accuracy verification
# ---------------------------------------------------------------------------

function order_table()
    dts = Float64[0.5, 0.25, 0.125, 0.0625, 0.03125]
    t_end = 4.0
    exact = exact_solution(t_end, X0_TEST)

    println("Stiff system: dx/dt = diag(-1, -1000) · x, exact at t_end = $t_end:")
    @printf("  exact = (%.6e, %.6e)\n\n", exact[1], exact[2])

    methods = [
        ("Backward Euler", simulate_be, 1),
        ("BDF2",           simulate_bdf2, 2),
    ]

    for (name, sim_fn, expected_order) in methods
        println("=" ^ 60)
        println("$name (expected order $expected_order)")
        println("-" ^ 60)
        errs = Float64[]
        for dt in dts
            x_T = sim_fn(A_STIFF, X0_TEST, dt, t_end)
            err = maximum(abs.(x_T .- exact))
            push!(errs, err)
            @printf("  dt = %8.5f   x_T = (%.6e, %.6e)   max-err = %.6e\n", dt, x_T[1], x_T[2], err)
        end
        slope = log(errs[end-1] / errs[end]) / log(dts[end-1] / dts[end])
        @printf("\n  empirical slope (finest two dt) ≈ %.3f   (expected %d)\n\n", slope, expected_order)
    end

    println("=" ^ 60)
    println("Interpretation:")
    println("  Both methods remain *stable* at every dt — no explosion even at")
    println("  dt = 0.5 (~ 500× the explicit-RK stability limit of 1/|λ_max| = 0.002).")
    println("  Empirical orders ≈ 1 (BE) and ≈ 2 (BDF2) confirm Ch 6 §6.2.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    println("Chapter 6 Julia companion: implicit methods on a stiff linear system")
    println("=" ^ 70)
    println()
    order_table()
end
