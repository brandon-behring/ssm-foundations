# runtests.jl --- Chapter 11 Julia companion test suite.
#
# Real `@test` assertions against fftconv.jl, pinning the §11.4 claims
# (`ch11:fftconv-causal`):
#   * the 2L-padded DFT convolution equals the explicit Toeplitz oracle < 1e-12;
#   * the convolution is causal (an impulse at t0 leaves earlier outputs at zero);
#   * the un-padded length-L cyclic convolution breaks causality (the 2L padding
#     is necessary).
#
# Module-isolation wrapper (per ch10 runtests) so includes do not collide.
#
# Usage:
#     julia --project=companions/ch11/julia companions/ch11/julia/runtests.jl

using Test
using LinearAlgebra
using Random

module Hyena
include("fftconv.jl")
end

function demo_inputs(; L = 64, D = 3, B = 2, seed = 0)
    # Deterministic inputs (stdlib Random with a fixed seed).
    rng = MersenneTwister(seed)
    u = randn(rng, B, L, D)
    taps = collect(0:(L - 1))
    k = randn(rng, D, L) .* exp.(-0.05 .* taps)'
    bias = randn(rng, D)
    return u, k, bias
end

@testset "Chapter 11 — fftconv.jl" begin

    @testset "§11.4 DFT conv == Toeplitz oracle (< 1e-12)" begin
        for L in (16, 64, 128)
            u, k, bias = demo_inputs(; L = L)
            y_dft = Hyena.dftconv(u, k, bias)
            y_naive = Hyena.causal_conv1d_naive(u, k, bias)
            @test maximum(abs.(y_dft .- y_naive)) < 1e-12
        end
    end

    @testset "causality: impulse at t0 leaves earlier outputs at zero" begin
        L, D, t0 = 48, 2, 30
        u = zeros(Float64, 1, L, D)
        u[1, t0, :] .= 1.0
        k = randn(MersenneTwister(1), D, L)
        bias = zeros(Float64, D)
        y = Hyena.dftconv(u, k, bias)
        @test maximum(abs.(y[1, 1:(t0 - 1), :])) < 1e-12
        # Response at/after t0 reproduces the kernel taps k[d, 1 : L-t0+1].
        for d in 1:D
            @test maximum(abs.(y[1, t0:L, d] .- k[d, 1:(L - t0 + 1)])) < 1e-12
        end
    end

    @testset "2L padding necessary: un-padded cyclic conv breaks causality" begin
        u, k, bias = demo_inputs(; L = 64)
        y_naive = Hyena.causal_conv1d_naive(u, k, bias)
        @test maximum(abs.(Hyena.dftconv(u, k, bias) .- y_naive)) < 1e-12
        @test maximum(abs.(Hyena.cyclic_conv_unpadded(u, k, bias) .- y_naive)) > 1e-1
    end

    @testset "argument validation" begin
        u, k, bias = demo_inputs(; L = 8, D = 2)
        @test_throws ArgumentError Hyena.dftconv(u, k[:, 1:(end - 1)], bias)   # wrong L
        @test_throws ArgumentError Hyena.dftconv(u, k, bias[1:(end - 1)])      # wrong D
    end
end
