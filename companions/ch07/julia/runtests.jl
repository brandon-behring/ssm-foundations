# runtests.jl --- Chapter 7 Julia companion test suite.
#
# Real `@test` assertions against the public functions of `hippo_legendre.jl`.
# Module-isolation wrapper per companions/ch04/julia/runtests.jl so future script
# includes do not collide on top-level names.
#
# Usage:
#     julia --project=companions/ch07/julia companions/ch07/julia/runtests.jl

using Test
using LinearAlgebra

module HippoLegendre
include("hippo_legendre.jl")
end

@testset "Chapter 7 — hippo_legendre.jl" begin
    @testset "Closed form matches the §7.3 oracle (entries + lower-triangular)" begin
        for n in (4, 6, 8)
            A, B = HippoLegendre.make_hippo_legs(n)
            Aexp = zeros(n, n)
            for i in 1:n, j in 1:n
                p, q = i - 1, j - 1
                if p > q
                    Aexp[i, j] = -sqrt((2p + 1) * (2q + 1))
                elseif p == q
                    Aexp[i, j] = -(p + 1)
                end
            end
            @test isapprox(A, Aexp; atol = 1e-12)
            @test isapprox(vec(B), [sqrt(2 * (i - 1) + 1) for i in 1:n]; atol = 1e-12)
            # strict upper triangle is exactly zero
            @test maximum(abs.(triu(A, 1))) == 0.0
        end
    end

    @testset "Spectrum: eigenvalues are exactly -1,...,-N (§7.7)" begin
        for n in (4, 8, 16, 32)
            eigs = HippoLegendre.legs_eigenvalues(n)
            @test maximum(abs.(imag.(eigs))) < 1e-8
            @test isapprox(sort(real(eigs)), sort(-(1.0:n) |> collect); atol = 1e-8)
            @test all(real.(eigs) .< 0.0)
        end
    end

    @testset "Legendre basis: orthonormal under the uniform measure on [0,1]" begin
        # ∫_0^1 P̃_i P̃_j dz ≈ δ_ij — a trapezoidal Gram-matrix check.
        n = 6
        z = range(0.0, 1.0; length = 2001) |> collect
        Pn = HippoLegendre.legendre_basis(n, z; normalized = true)
        dz = z[2] - z[1]
        # Proper trapezoidal weights (halved endpoints); a plain rectangle sum
        # over-counts the large Legendre endpoint values P̃_n(0,1)=±sqrt(2n+1).
        w = fill(dz, length(z))
        w[1] = dz / 2
        w[end] = dz / 2
        gram = (Pn .* w') * Pn'
        @test isapprox(gram, Matrix{Float64}(I, n, n); atol = 1e-3)
    end

    @testset "Online reconstruction error decreases with N (§7.1)" begin
        e4 = HippoLegendre.reconstruction_error(4)
        e8 = HippoLegendre.reconstruction_error(8)
        e16 = HippoLegendre.reconstruction_error(16)
        e64 = HippoLegendre.reconstruction_error(64)
        @test e8 < e4
        @test e16 < e8
        @test e16 < 0.05      # N=16 captures the two-sinusoid signal
        @test e64 < 0.01      # reaches the discretization floor
    end

    @testset "Encoder matches an independent naive recompute" begin
        n, L = 8, 60
        z = range(0.0, 1.0; length = L) |> collect
        u = sin.(2π .* 2.0 .* z)
        got = HippoLegendre.hippo_legs_encode(u, n)

        A, B = HippoLegendre.make_hippo_legs(n)
        Apos = -A
        Bv = vec(B)
        eyeN = Matrix{Float64}(I, n, n)
        c = zeros(n)
        ref = zeros(L, n)
        for k in 1:L
            c = (eyeN + Apos / (2k)) \ ((eyeN - Apos / (2k)) * c + (Bv / k) * u[k])
            ref[k, :] = c
        end
        @test isapprox(got, ref; atol = 1e-10)
    end

    @testset "Input validation (no silent failure)" begin
        @test_throws ArgumentError HippoLegendre.make_hippo_legs(0)
    end
end
