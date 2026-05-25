# runtests.jl --- Chapter 6 Julia companion test suite.
#
# Real `@test` assertions for both `implicit_methods.jl` and
# `symplectic_methods.jl`. Verifies the load-bearing pedagogical claims of
# Chapter 6: implicit-Euler L-stability on stiff systems, BDF2 second-order
# convergence, and the symplectic-vs-non-symplectic energy-drift contrast.
#
# Usage:
#     julia --project=companions/ch06/julia companions/ch06/julia/runtests.jl

using Test
using LinearAlgebra

module Implicit
include("implicit_methods.jl")
end

module Symplectic
include("symplectic_methods.jl")
end

@testset "Chapter 6 — implicit_methods.jl" begin
    A = Implicit.A_STIFF
    x0 = Implicit.X0_TEST
    t_end = 4.0
    exact = Implicit.exact_solution(t_end, x0)

    @testset "Backward Euler remains stable at dt >> explicit-RK limit" begin
        # Explicit RK on a -1000 eigenvalue needs dt < 2e-3; verify backward
        # Euler returns a bounded solution at dt = 0.5 (250× the explicit limit).
        x_T = Implicit.simulate_be(A, x0, 0.5, t_end)
        @test all(isfinite, x_T)
        @test maximum(abs.(x_T)) < 10.0  # not exploding
    end

    @testset "Backward Euler converges to exact solution as dt -> 0" begin
        errs = Float64[]
        for dt in (0.5, 0.25, 0.125)
            x_T = Implicit.simulate_be(A, x0, dt, t_end)
            push!(errs, maximum(abs.(x_T .- exact)))
        end
        # First-order method: halving dt should roughly halve error.
        @test errs[2] < 0.7 * errs[1]
        @test errs[3] < 0.7 * errs[2]
    end

    @testset "BDF2 empirical slope ≈ 2 (Ch 6 §6.2 claim)" begin
        # Verify the slope from the two finest step sizes is close to 2.
        dts = (0.125, 0.0625)
        errs = Float64[]
        for dt in dts
            x_T = Implicit.simulate_bdf2(A, x0, dt, t_end)
            push!(errs, maximum(abs.(x_T .- exact)))
        end
        slope = log(errs[1] / errs[2]) / log(dts[1] / dts[2])
        @test slope > 1.5  # close to expected 2; allow slack for stiff problem
    end
end

@testset "Chapter 6 — symplectic_methods.jl" begin
    q0, p0 = 1.0, 0.0
    E0 = Symplectic.H_energy(q0, p0)

    @testset "Verlet preserves energy within bounded oscillation over 100 periods" begin
        # The defining symplectic-vs-non-symplectic contrast (Ch 6 §6.4 +
        # the energy_drift.png figure caption in ch06-implicit-and-symplectic.mdx).
        dt = 0.05
        n_steps = Int(round(100 * 2π / dt))
        _, _, energies = Symplectic.simulate(Symplectic.verlet_step, q0, p0, dt, n_steps)
        drift = maximum(energies) - minimum(energies)
        # Bounded by O(dt^2) ~ 0.05^2 = 2.5e-3, allow generous slack.
        @test drift < 1e-2
        # Final energy stays close to initial (no secular drift).
        @test abs(energies[end] - E0) < 1e-2
    end

    @testset "RK4 energy drift is monotonic; symplectic methods oscillate (the qualitative contrast)" begin
        # The pedagogical distinction at the heart of Ch 6: non-symplectic
        # methods accumulate energy secularly (monotonic drift), while
        # symplectic methods produce bounded oscillation. The ABSOLUTE
        # magnitude depends on (dt, horizon, method order) and isn't the
        # pedagogically load-bearing claim. The QUALITATIVE distinction is.
        dt = 0.05
        n_steps = Int(round(100 * 2π / dt))

        _, _, sympE = Symplectic.simulate(
            Symplectic.symplectic_euler_step, q0, p0, dt, n_steps,
        )
        _, _, rk4E = Symplectic.simulate(Symplectic.rk4_step, q0, p0, dt, n_steps)

        # Symplectic Euler: energy oscillates around E0 — both E_max > E0 and
        # E_min < E0 are visited.
        @test maximum(sympE) > E0
        @test minimum(sympE) < E0

        # RK4: energy drift is monotonic (one-sided). E_min < E0 and
        # E_max ≈ E0 (never goes above) — or vice versa. The hallmark of
        # secular accumulation.
        rk4_dev_pos = maximum(rk4E) - E0
        rk4_dev_neg = E0 - minimum(rk4E)
        # One side of the deviation should dwarf the other for a monotonic
        # drift; symplectic methods show the two sides comparable.
        @test min(rk4_dev_pos, rk4_dev_neg) < 1e-9 * max(rk4_dev_pos, rk4_dev_neg) ||
              min(rk4_dev_pos, rk4_dev_neg) == 0.0
    end

    @testset "Verlet achieves order-2 empirical accuracy on state error" begin
        # Order on the state at t = 1.5 (mid-orbit so phase error doesn't
        # contaminate amplitude error). Verify slope ≈ 2.
        t_end = 1.5
        q_exact = cos(t_end) * q0 + sin(t_end) * p0
        p_exact = -sin(t_end) * q0 + cos(t_end) * p0
        errs = Float64[]
        dts = (0.025, 0.0125)
        for dt in dts
            n_steps = Int(round(t_end / dt))
            q, p, _ = Symplectic.simulate(Symplectic.verlet_step, q0, p0, dt, n_steps)
            push!(errs, max(abs(q - q_exact), abs(p - p_exact)))
        end
        slope = log(errs[1] / errs[2]) / log(dts[1] / dts[2])
        @test slope > 1.7  # close to expected 2
    end
end
