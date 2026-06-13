# runtests.jl --- Chapter 17 Julia companion test suite.
#
# Real `@test` assertions against symplectic_crosscheck.jl, pinning the §17.2 C1 atlas-cell facts
# cross-language to the SAME literal values the JAX companion produces (the cross-language anchors,
# computed from identical deterministic integrator arithmetic — no RNG):
#   * the exact-exponential (diagonal SSM) transition conserves the imaginary-mode energy;
#   * Störmer-Verlet has a near-zero secular slope and a bounded energy band == JAX's;
#   * RK4's endpoint energy drift == JAX's (and == ch06's rk4_drift_per_period).
#
# Module-isolation wrapper (per ch10-ch15 runtests) so includes do not collide.
#
# Usage:
#     julia --project=companions/ch17/julia companions/ch17/julia/runtests.jl

using Test

module Ch17Symplectic
include("symplectic_crosscheck.jl")
end

const DT = 0.1
const PERIODS = 200
# Cross-language anchors: literals from companions/ch17/jax/c1_integration.py (dt=0.1, 200 periods).
const JAX_RK4_ENDPOINT = -4.357489219565411e-7
const JAX_VERLET_BAND = 0.001249999967944282

@testset "Chapter 17 C1 symplectic cross-check" begin
    n = Ch17Symplectic.n_steps(DT, PERIODS)
    e_exp = Ch17Symplectic.exact_exp_energy(1.0, 0.0, DT, n)
    e_ver = Ch17Symplectic.energy_trajectory(Ch17Symplectic.verlet_step, 1.0, 0.0, DT, n)
    e_rk4 = Ch17Symplectic.energy_trajectory(Ch17Symplectic.rk4_step, 1.0, 0.0, DT, n)

    @testset "exact-exponential conserves (diagonal SSM)" begin
        @test Ch17Symplectic.energy_band(e_exp) < 1e-9          # JAX 2.8e-12
    end

    @testset "symplectic: zero secular drift, bounded band == JAX" begin
        @test abs(Ch17Symplectic.secular_slope_per_period(e_ver, DT)) < 1e-6   # JAX 5.8e-9
        @test isapprox(Ch17Symplectic.energy_band(e_ver), JAX_VERLET_BAND; atol = 1e-7)
    end

    @testset "RK4 endpoint drift == JAX (cross-language anchor)" begin
        @test isapprox(Ch17Symplectic.endpoint_drift_per_period(e_rk4, DT), JAX_RK4_ENDPOINT; atol = 1e-9)
        # the symplectic integrator kills the secular trend RK4 carries
        @test abs(Ch17Symplectic.secular_slope_per_period(e_ver, DT)) <
              abs(Ch17Symplectic.secular_slope_per_period(e_rk4, DT))
    end

    @testset "validation" begin
        @test_throws ArgumentError Ch17Symplectic.n_steps(0.0, 10)
        @test_throws ArgumentError Ch17Symplectic.n_steps(0.1, 0)
    end
end
