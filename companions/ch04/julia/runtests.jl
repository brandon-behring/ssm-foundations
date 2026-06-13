# runtests.jl --- Chapter 4 Julia companion test suite.
#
# Real `@test` assertions (not smoke wraps) against the public functions of
# `discretization_atlas.jl`. Module-isolation wrapper per
# `post_transformers/experiments/julia/test/runtests.jl:29-31`, so future
# additional script includes do not collide on top-level `const`s.
#
# Usage:
#     julia --project=companions/ch04/julia companions/ch04/julia/runtests.jl

using Test
using LinearAlgebra

module DiscretizationAtlas
include("discretization_atlas.jl")
end

@testset "Chapter 4 — discretization_atlas.jl" begin
    A = DiscretizationAtlas.A_OSC
    B = DiscretizationAtlas.B_OSC

    @testset "ZOH autonomous-exact: Ad == exp(A dt) when forcing is absorbed correctly" begin
        # The augmented-matrix trick produces Ad = exp(A dt) for the homogeneous
        # part regardless of B. Verify the homogeneous discretization matches
        # the direct matrix exponential within machine tolerance.
        dt = 0.1
        result = DiscretizationAtlas.discretize_zoh(A, B, dt)
        expected_Ad = exp(A .* dt)
        @test isapprox(result.Ad, expected_Ad; atol = 1e-12, rtol = 1e-12)
    end

    @testset "Bilinear (Tustin) gives a distinct Ad from ZOH on this oscillator" begin
        # Sanity check that the bilinear formula does not accidentally collapse
        # to the ZOH formula. (They agree only in the dt -> 0 limit.)
        dt = 0.1
        zoh = DiscretizationAtlas.discretize_zoh(A, B, dt)
        bil = DiscretizationAtlas.discretize_bilinear(A, B, dt)
        @test !isapprox(zoh.Ad, bil.Ad; atol = 1e-3)
    end

    @testset "Bilinear preserves stability (eigenvalues of Ad inside unit disk)" begin
        # A has eigenvalues -0.25 ± i√15.75/2 ≈ -0.25 ± 1.984i (open left half-plane). The bilinear
        # transform maps the open LHP to the open unit disk, so |λ(Ad)| < 1
        # at any dt > 0.
        for dt in (0.01, 0.1, 0.5, 1.0)
            params = DiscretizationAtlas.discretize_bilinear(A, B, dt)
            spec_rad = maximum(abs.(eigvals(params.Ad)))
            @test spec_rad < 1.0
        end
    end

    @testset "Forced simulation converges to Tsit5 reference as dt -> 0" begin
        # Empirical first-order accuracy for ZOH on the forced system,
        # matching Chapter 4 Table 4.1.
        t_end = 4.0
        ref = DiscretizationAtlas.continuous_forced(t_end)
        errs = Float64[]
        dts = Float64[0.2, 0.1, 0.05]
        for dt in dts
            ts, ys = DiscretizationAtlas.simulate(
                DiscretizationAtlas.discretize_zoh,
                DiscretizationAtlas.step_zoh,
                dt,
                t_end,
            )
            yref = [dot(DiscretizationAtlas.C_OSC, ref(t)) for t in ts]
            push!(errs, maximum(abs.(ys .- yref)))
        end
        # Convergence: each halving of dt should reduce error by a factor near 2
        # for first-order ZOH. Allow generous slack.
        @test errs[2] < 0.7 * errs[1]
        @test errs[3] < 0.7 * errs[2]
    end

    @testset "Exp-trapezoidal achieves order >= 2 (F29 regression guard)" begin
        # Empirical second-order accuracy for exp-trap on the forced oscillator.
        # Before the F29 fix, M[n+1, n+2] = T(1.0) made B1 too small by a factor
        # of dt, degrading exp-trap to first order — halving dt would have given
        # ratio ~0.5 (matching ZOH). After the fix, ratio is ~0.25.
        t_end = 4.0
        ref = DiscretizationAtlas.continuous_forced(t_end)
        errs = Float64[]
        dts = Float64[0.2, 0.1, 0.05]
        for dt in dts
            ts, ys = DiscretizationAtlas.simulate(
                DiscretizationAtlas.discretize_exp_trap,
                DiscretizationAtlas.step_exp_trap,
                dt,
                t_end,
            )
            yref = [dot(DiscretizationAtlas.C_OSC, ref(t)) for t in ts]
            push!(errs, maximum(abs.(ys .- yref)))
        end
        # Second-order: halving dt should reduce error by a factor near 4.
        # The 0.35 threshold separates 2nd-order (ratio ~0.25) from 1st-order
        # (ratio ~0.5), with slack for finite-dt effects.
        @test errs[2] < 0.35 * errs[1]
        @test errs[3] < 0.35 * errs[2]
    end
end
