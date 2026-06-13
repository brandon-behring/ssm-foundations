# runtests.jl --- Chapter 15 Julia companion test suite.
#
# Real `@test` assertions against lyapunov_crosscheck.jl, pinning the §15.4-15.5
# facts cross-language to the SAME literal values the JAX suite produces (the
# cross-language numeric anchors):
#   * diagonal LTI spectrum recovered exactly == JAX-pinned values (< 1e-9);
#   * Chapter 13 DPLR closed-form spectrum == JAX (< 1e-9); QR estimate matches its
#     closed form to the O(1/T) tolerance; divergence identity sum = log|det|;
#   * effective state size of [1]^3 || [0.4]^5 == the closed form == JAX.
#
# Module-isolation wrapper (per ch10-ch13 runtests) so includes do not collide.
#
# Usage:
#     julia --project=companions/ch15/julia companions/ch15/julia/runtests.jl

using Test
using LinearAlgebra

module Ch15Lyapunov
include("lyapunov_crosscheck.jl")
end

# Cross-language anchors: literals identical to companions/ch15/jax/lyapunov_diagnostics.py.
const DIAG_VALS = [0.9, 0.7, 0.5, 0.3]
const DIAG_SPECTRUM = [-0.105360515658, -0.356674943939, -0.693147180560, -1.203972804326]
const DPLR_W = [0.9, 0.8, 0.7, 0.6, 0.5]
const DPLR_C = 0.2
const DPLR_CLOSED = [-0.128148601305, -0.255109480374, -0.400693038579,
                     -0.575556228685, -0.883828184846]
const DEFF_3_8_04 = 4.616368286445013

@testset "Chapter 15 Julia cross-check" begin
    @testset "diagonal LTI: exact recovery == JAX" begin
        J = Matrix(Diagonal(DIAG_VALS))
        spec = Ch15Lyapunov.qr_lyapunov([J], 2000)
        # diagonal -> QR is trivial -> recovered exactly, matching the JAX values
        @test isapprox(spec, DIAG_SPECTRUM; atol = 1e-9)
        @test isapprox(spec, Ch15Lyapunov.closed_form_log_growth(J); atol = 1e-12)
    end

    @testset "DPLR (Ch 13): closed form == JAX, QR recovers it" begin
        a = [1.0, -1.0, 1.0, -1.0, 1.0]
        a = a ./ norm(a)
        J = Ch15Lyapunov.dplr_transition(DPLR_W, a, DPLR_C)
        @test issymmetric(J)
        closed = Ch15Lyapunov.closed_form_log_growth(J)
        @test isapprox(closed, DPLR_CLOSED; atol = 1e-9)        # cross-language eigendecomposition
        est = Ch15Lyapunov.qr_lyapunov([J], 4000)
        @test isapprox(est, closed; atol = 1e-3)                # QR estimate recovers it (O(1/T))
        # divergence identity: sum(lambda) = log|det J| (autonomous)
        @test isapprox(sum(est), log(abs(det(J))); atol = 1e-9)
    end

    @testset "effective state size (P3'): two-level spectrum == closed form == JAX" begin
        mags = vcat(ones(3), fill(0.4, 5))   # [1]^3 || [0.4]^5, d = 8, r = 3
        deff = Ch15Lyapunov.effective_state_size(mags)
        @test isapprox(deff, Ch15Lyapunov.effective_state_size_closed_form(3, 8, 0.4); atol = 1e-12)
        @test isapprox(deff, DEFF_3_8_04; atol = 1e-9)
        # limits: w -> 0 gives r, w = 1 gives d
        @test Ch15Lyapunov.effective_state_size_closed_form(3, 8, 0.0) == 3.0
        @test Ch15Lyapunov.effective_state_size_closed_form(3, 8, 1.0) == 8.0
    end

    @testset "validation" begin
        @test_throws ArgumentError Ch15Lyapunov.qr_lyapunov([Matrix(Diagonal(DIAG_VALS))], 0)
        @test_throws ArgumentError Ch15Lyapunov.effective_state_size(zeros(4))
    end
end
