# runtests.jl --- Chapter 5 Julia companion test suite.
#
# Real `@test` assertions against the public functions of
# `butcher_tableau_zoo.jl`. Verifies Butcher order conditions and the
# correspondence between stability-function Taylor coefficients and exp(z)
# Taylor coefficients (the algebraic content of Ch 5 §5.3).
#
# Usage:
#     julia --project=companions/ch05/julia companions/ch05/julia/runtests.jl

using Test

module ButcherZoo
include("butcher_tableau_zoo.jl")
end

@testset "Chapter 5 — butcher_tableau_zoo.jl" begin
    methods = [
        ButcherZoo.forward_euler(),
        ButcherZoo.midpoint_rk2(),
        ButcherZoo.heun_rk2(),
        ButcherZoo.classical_rk4(),
        ButcherZoo.rkf45(),
    ]

    @testset "Order-1 condition: ∑ b_i == 1 (all methods)" begin
        for tab in methods
            oc = ButcherZoo.verify_order_conditions(tab)
            @test isapprox(oc.sum1, 1.0; atol = 1e-12)
        end
    end

    @testset "Order-2 condition: ∑ b_i c_i == 1/2 (order ≥ 2 methods only)" begin
        for tab in methods
            oc = ButcherZoo.verify_order_conditions(tab)
            if tab.expected_order >= 2
                @test isapprox(oc.sum2, 0.5; atol = 1e-12)
            end
        end
        # Forward Euler must NOT satisfy order-2 (sum2 = 0 because c_1 = 0).
        oc = ButcherZoo.verify_order_conditions(ButcherZoo.forward_euler())
        @test !isapprox(oc.sum2, 0.5; atol = 1e-3)
    end

    @testset "Order-3 conditions: ∑ b_i c_i² == 1/3 and ∑ b_i a_ij c_j == 1/6" begin
        for tab in methods
            oc = ButcherZoo.verify_order_conditions(tab)
            if tab.expected_order >= 3
                @test isapprox(oc.sum3, 1 / 3; atol = 1e-12)
                @test isapprox(oc.sum4, 1 / 6; atol = 1e-12)
            end
        end
    end

    @testset "Stability function R(z) matches exp(z) Taylor through method order" begin
        # An order-p method has R(z) agreeing with exp(z) Taylor coefficients
        # through degree p (this IS the algebraic content of the order
        # conditions, viewed from the stability-function side).
        for tab in methods
            coeffs = ButcherZoo.stability_taylor(tab; degree = 6)
            expected = ButcherZoo.expected_exp_coeffs(6)
            for k in 0:tab.expected_order
                @test isapprox(coeffs[k + 1], expected[k + 1]; atol = 1e-12)
            end
        end
    end

    @testset "Stability function R(z) DIFFERS from exp(z) above method order" begin
        # Conversely, order-p methods must differ from exp(z) at coefficient
        # k = p+1 (otherwise they would be order p+1). Skip RKF (order 5,
        # degree-6 difference is tiny but nonzero for Fehlberg's tableau).
        for tab in (ButcherZoo.forward_euler(),
                    ButcherZoo.midpoint_rk2(),
                    ButcherZoo.classical_rk4())
            coeffs = ButcherZoo.stability_taylor(tab; degree = tab.expected_order + 2)
            expected = ButcherZoo.expected_exp_coeffs(tab.expected_order + 2)
            k = tab.expected_order + 1
            @test !isapprox(coeffs[k + 1], expected[k + 1]; atol = 1e-6)
        end
    end
end
