# runtests.jl --- Chapter 13 Julia companion test suite.
#
# Real `@test` assertions against xlstm_stabilization.jl, pinning the §13.4
# claims cross-language (the same facts the JAX suite pins):
#   * P2 stabilizer exactness: stabilized == naive in the safe regime, < 1e-12;
#   * rescaled gates f', i' in (0, 1];
#   * overflow: naive non-finite at log_i = 760, stabilized finite;
#   * single-pair recovery = [0.3, -0.7, 1.1] at log_i = 800 (the cross-language
#     numeric anchor shared with the JAX companion).
#
# Module-isolation wrapper (per ch10-ch12 runtests) so includes do not collide.
#
# Usage:
#     julia --project=companions/ch13/julia companions/ch13/julia/runtests.jl

using Test
using LinearAlgebra
using Random

module XLSTM
include("xlstm_stabilization.jl")
end

function safe_stream(rng; L = 24, d_k = 6, d_v = 5)
    q = randn(rng, L, d_k)
    k = randn(rng, L, d_k)
    k = k ./ mapslices(norm, k; dims = 2)
    v = randn(rng, L, d_v)
    log_f = XLSTM.log_sigmoid.(rand(rng, L) .* 2.0)  # log σ of uniform(0, 2) preacts
    log_i = (rand(rng, L) .* 4.0) .- 2.0             # uniform(-2, 2)
    return q, k, v, log_f, log_i
end

@testset "Chapter 13 — xlstm_stabilization.jl" begin

    @testset "§13.4 P2: stabilized == naive in the safe regime (< 1e-12)" begin
        for seed in (0, 1, 7)
            rng = MersenneTwister(seed)
            q, k, v, log_f, log_i = safe_stream(rng)
            Hn = XLSTM.mlstm_naive(q, k, v, log_f, log_i)
            Hs, _ = XLSTM.mlstm_stabilized(q, k, v, log_f, log_i)
            @test all(isfinite, Hn)
            @test maximum(abs.(Hn .- Hs)) < 1e-12
        end
    end

    @testset "§13.4 rescaled gates in (0, 1]" begin
        rng = MersenneTwister(3)
        q, k, v, log_f, log_i = safe_stream(rng)
        _, mtraj = XLSTM.mlstm_stabilized(q, k, v, log_f, log_i)
        mprev = vcat(-Inf, mtraj[1:(end - 1)])
        @test all(log_i .<= mtraj .+ 1e-12)            # i'_t = exp(log_i - m_t) <= 1
        @test all(log_f .+ mprev .<= mtraj .+ 1e-12)   # f'_t = exp(log_f + m_{t-1} - m_t) <= 1
    end

    @testset "§13.4 overflow: naive dies, stabilized finite" begin
        rng = MersenneTwister(0)
        L, d_k, d_v = 16, 4, 3
        q = randn(rng, L, d_k)
        k = randn(rng, L, d_k)
        k = k ./ mapslices(norm, k; dims = 2)
        v = randn(rng, L, d_v)
        log_f = XLSTM.log_sigmoid.(rand(rng, L) .* 2.0)
        log_i = (rand(rng, L) .* 2.0) .- 1.0
        log_i[L ÷ 2] = 760.0
        Hn = XLSTM.mlstm_naive(q, k, v, log_f, log_i)
        Hs, _ = XLSTM.mlstm_stabilized(q, k, v, log_f, log_i)
        @test !all(isfinite, Hn)
        @test all(isfinite, Hs)
    end

    @testset "§13.4 single-pair recovery = v at any gate (cross-language anchor)" begin
        for log_i in (0.0, 50.0, 800.0)
            err, _ = XLSTM.single_pair_recovery(log_i)
            @test err < 1e-12
        end
        _, readout = XLSTM.single_pair_recovery(800.0)
        @test maximum(abs.(readout .- [0.3, -0.7, 1.1])) < 1e-12
    end

    @testset "log_sigmoid <= 0 and matches the definition" begin
        for x in range(-10.0, 10.0; length = 21)
            @test XLSTM.log_sigmoid(x) ≈ log(1.0 / (1.0 + exp(-x))) atol = 1e-12
            @test XLSTM.log_sigmoid(x) <= 0.0
        end
    end

    @testset "argument validation" begin
        rng = MersenneTwister(0)
        q, k, v, log_f, log_i = safe_stream(rng)
        @test_throws ArgumentError XLSTM.mlstm_naive(q, k, v, log_f[1:(end - 1)], log_i)
        @test_throws ArgumentError XLSTM.mlstm_stabilized(q, k, v[1:(end - 1), :], log_f, log_i)
    end
end
