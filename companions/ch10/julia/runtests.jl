# runtests.jl --- Chapter 10 Julia companion test suite.
#
# Real `@test` assertions against the public functions of `discretization.jl`,
# pinning the same §10.2-10.3 claims as the JAX suite:
#   * exp-trapezoidal is second-order on a FORCED system; ZOH is first-order;
#   * on the HOMOGENEOUS system both exponential schemes are exact (the order is
#     invisible) — the chapter's load-bearing subtlety;
#   * exponential schemes damp stiff modes (α → 0); bilinear does not (α → -1);
#     forward Euler is not A-stable.
#
# Module-isolation wrapper (per ch04/ch07 runtests) so future includes do not
# collide on top-level names.
#
# Usage:
#     julia --project=companions/ch10/julia companions/ch10/julia/runtests.jl

using Test
using LinearAlgebra

module Disc
include("discretization.jl")
end

const A = -0.5 + 2.0im      # one complex decaying-oscillating mode
const ω = 1.3
const T_END = 6.0
const X0 = 1.0 + 0.0im
const DTS = [0.2, 0.1, 0.05, 0.025, 0.0125]

@testset "Chapter 10 — discretization.jl" begin

    @testset "§10.2 order of accuracy (forced)" begin
        _, s_zoh = Disc.order_sweep(:zoh, A, DTS, T_END, ω; x0 = X0)
        _, s_trap = Disc.order_sweep(:exp_trapezoidal, A, DTS, T_END, ω; x0 = X0)
        _, s_bl = Disc.order_sweep(:bilinear, A, DTS, T_END, ω; x0 = X0)
        @test 0.9 < s_zoh < 1.15      # first-order
        @test 1.9 < s_trap < 2.1      # second-order
        @test 1.9 < s_bl < 2.1        # second-order
    end

    @testset "exp-trapezoidal beats ZOH at fixed dt (forced)" begin
        e_zoh = Disc.global_error(:zoh, A, 0.05, T_END, ω; x0 = X0)
        e_trap = Disc.global_error(:exp_trapezoidal, A, 0.05, T_END, ω; x0 = X0)
        @test e_trap < e_zoh
    end

    @testset "§10.2 homogeneous-blindness: order invisible on u ≡ 0" begin
        # Both exponential schemes are exact on the autonomous system, regardless
        # of their order. This is why Mamba-3's order-2 claim needs a forced test.
        @test Disc.homogeneous_error(:zoh, A, 0.1, T_END; x0 = X0) < 1e-12
        @test Disc.homogeneous_error(:exp_trapezoidal, A, 0.1, T_END; x0 = X0) < 1e-12
    end

    @testset "homogeneous transition coefficients are identical (both e^{A dt})" begin
        αz, _ = Disc.discretize_zoh(A, 0.1)
        αt, _, _ = Disc.discretize_exp_trapezoidal(A, 0.1)
        @test αz == αt
    end

    @testset "coefficient identities" begin
        # λ = 1 degenerates to a shifted ZOH: β = 0, γ = dt.
        _, β1, γ1 = Disc.discretize_exp_trapezoidal(A, 0.1; λ = 1.0)
        @test abs(β1) < 1e-15
        @test isapprox(γ1, 0.1; atol = 1e-15)
        # λ = 1/2: γ = dt/2 and β = (dt/2)·α.
        α, β, γ = Disc.discretize_exp_trapezoidal(A, 0.1; λ = 0.5)
        @test isapprox(γ, 0.05; atol = 1e-15)
        @test isapprox(β, 0.05 * α; atol = 1e-15)
    end

    @testset "schemes agree as dt → 0" begin
        dt = 1e-6
        αz, _ = Disc.discretize_zoh(A, dt)
        αt, _, _ = Disc.discretize_exp_trapezoidal(A, dt)
        αb, _ = Disc.discretize_bilinear(A, dt)
        @test isapprox(αz, αt; atol = 1e-10)
        @test isapprox(αb, αz; atol = 1e-10)
    end

    @testset "ZOH β limit at A → 0 is dt (L'Hôpital guard)" begin
        _, β = Disc.discretize_zoh(0.0 + 0.0im, 0.3)
        @test isapprox(real(β), 0.3; atol = 1e-12)
    end

    @testset "§10.3 stability" begin
        # Exponential schemes A-stable over the whole left half-plane.
        for _ in 1:200
            z = -50 * rand() + 1im * (rand() - 0.5) * 100
            @test abs(Disc.amplification(:exp_trapezoidal, z)) <= 1 + 1e-12
        end
        # Stiff-mode contrast.
        z = -50.0
        @test abs(Disc.amplification(:exp_trapezoidal, z)) < 1e-12   # → 0
        @test abs(Disc.amplification(:bilinear, z)) > 0.9            # → 1
        @test abs(Disc.amplification(:forward_euler, z)) > 1.0       # unstable
    end

    @testset "forward Euler not A-stable (Re z < 0 but outside disk)" begin
        @test abs(Disc.amplification(:forward_euler, -3.0)) > 1.0
    end

    @testset "argument validation" begin
        @test_throws ArgumentError Disc.discretize_exp_trapezoidal(A, 0.1; λ = 1.5)
        @test_throws ArgumentError Disc.amplification(:midpoint, -1.0)
        @test_throws ArgumentError Disc.integrate(:midpoint, A, 0.1, 4, ω)
    end
end
