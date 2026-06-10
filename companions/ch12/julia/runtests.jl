# runtests.jl --- Chapter 12 Julia companion test suite.
#
# Real `@test` assertions against delta_stability.jl, pinning the §12.4 claims
# cross-language (the same numbers the JAX suite pins):
#   * analytic spectral radius == Rayleigh quotient of the materialised
#     iteration matrix, < 1e-12 across the parameter grid;
#   * DeltaNet stable exactly on beta*||k||^2 in (0, 2); boundary constant 2;
#   * Longhorn rho = alpha/(alpha + ||k||^2) < 1 always, complement identity
#     rho + beta_eff*||k||^2 = 1;
#   * structural identity: longhorn_step == delta_rule_step at beta_eff;
#   * exact geometric error decay with the analytic ratio (1.5 divergence past
#     the boundary; 0.5 contraction for alpha = ||k||^2 = 1).
#
# Module-isolation wrapper (per ch10/ch11 runtests) so includes do not collide.
#
# Usage:
#     julia --project=companions/ch12/julia companions/ch12/julia/runtests.jl

using Test
using LinearAlgebra
using Random

module DeltaLineage
include("delta_stability.jl")
end

@testset "Chapter 12 — delta_stability.jl" begin

    @testset "§12.4 analytic rho == Rayleigh eigenvalue (< 1e-12)" begin
        rng = MersenneTwister(0)
        for bk in range(0.05, 2.95; length = 25)
            k = randn(rng, 8)
            beta_eff = bk / dot(k, k)
            lam = DeltaLineage.iteration_eigenvalue_along_k(k, beta_eff)
            @test abs(lam - (1.0 - bk)) < 1e-12
        end
    end

    @testset "§12.4 DeltaNet stability interval (0, 2)" begin
        @test DeltaLineage.a_stability_boundary() == 2.0
        for bk in range(0.05, 3.0; length = 60)
            rho = DeltaLineage.deltanet_spectral_radius(bk, 1.0)
            if bk < 2.0
                @test rho < 1.0
            else
                @test rho >= 1.0
            end
        end
        @test DeltaLineage.deltanet_spectral_radius(2.0, 1.0) == 1.0
    end

    @testset "§12.4 Longhorn unconditional stability + complement identity" begin
        for alpha in (1e-3, 0.1, 1.0, 10.0), ksq in exp10.(range(-4, 8; length = 30))
            rho = DeltaLineage.longhorn_spectral_radius(alpha, ksq)
            @test 0.0 < rho < 1.0
            beta_eff = DeltaLineage.longhorn_effective_beta(sqrt(ksq) .* [1.0], alpha)
            @test abs(rho + beta_eff * ksq - 1.0) < 1e-12
        end
    end

    @testset "§12.3 structural identity: Longhorn == delta rule at beta_eff" begin
        rng = MersenneTwister(2)
        S = randn(rng, 6, 8)
        k = randn(rng, 8)
        v = randn(rng, 6)
        for alpha in (0.1, 0.7, 5.0)
            lh = DeltaLineage.longhorn_step(S, k, v, alpha)
            dn = DeltaLineage.delta_rule_step(S, k, v,
                                              DeltaLineage.longhorn_effective_beta(k, alpha))
            @test maximum(abs.(lh .- dn)) < 1e-15
        end
    end

    @testset "§12.3 closed form == dense implicit solve (independent oracle)" begin
        # Solve the stationarity system S_t (alpha*I + k*k') = alpha*S + v*k'
        # directly — no shared code with the rank-one closed form.
        rng = MersenneTwister(4)
        S = randn(rng, 6, 8)
        k = randn(rng, 8)
        v = randn(rng, 6)
        for alpha in (0.1, 0.7, 5.0)
            closed = DeltaLineage.longhorn_step(S, k, v, alpha)
            solved = (alpha .* S .+ v * k') / (alpha * I + k * k')
            @test maximum(abs.(closed .- solved)) < 1e-12
            residual = alpha .* (closed .- S) .+ (closed * k .- v) * k'
            @test maximum(abs.(residual)) < 1e-12
        end
    end

    @testset "§12.1 fixed point invariant under any beta" begin
        rng = MersenneTwister(3)
        k = randn(rng, 8)
        v = randn(rng, 6)
        Sstar = DeltaLineage.delta_rule_fixed_point(k, v)
        for beta in (0.1, 1.0, 2.5)
            @test maximum(abs.(DeltaLineage.delta_rule_step(Sstar, k, v, beta) .- Sstar)) < 1e-12
        end
    end

    @testset "§12.4 exact geometric decay with the analytic ratio" begin
        rng = MersenneTwister(0)
        k = randn(rng, 8)
        k = k / norm(k)   # unit key: beta * ||k||^2 = beta
        v = randn(rng, 6)
        for (kwargs, rho) in (((beta = 0.5,), 0.5), ((beta = 2.5,), 1.5), ((alpha = 1.0,), 0.5))
            traj = DeltaLineage.error_trajectory(k, v, 12; kwargs...)
            # Early steps sit far above the float noise floor: pin at 1e-12.
            for t in 1:3
                @test abs(traj[t + 1] / traj[t] - rho) < 1e-12
            end
            # Restrict the full-trajectory pin to steps above the noise floor.
            for t in 1:12
                traj[t] >= 1e-4 * traj[1] || continue
                @test abs(traj[t + 1] / traj[t] - rho) < 1e-10
            end
        end
    end

    @testset "argument validation" begin
        @test_throws ArgumentError DeltaLineage.delta_rule_fixed_point(zeros(8), ones(6))
        @test_throws ArgumentError DeltaLineage.longhorn_effective_beta(ones(8), 0.0)
        @test_throws ArgumentError DeltaLineage.iteration_eigenvalue_along_k(zeros(8), 0.5)
        @test_throws ArgumentError DeltaLineage.delta_rule_step(zeros(6, 7), ones(8), ones(6), 0.5)
        @test_throws ArgumentError DeltaLineage.error_trajectory(ones(4), ones(3), 5)
        @test_throws ArgumentError DeltaLineage.error_trajectory(ones(4), ones(3), 5;
                                                                 beta = 0.5, alpha = 1.0)
    end
end
