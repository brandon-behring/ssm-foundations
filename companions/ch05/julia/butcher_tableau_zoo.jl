# butcher_tableau_zoo.jl
#
# Chapter 5 Julia companion: Butcher tableaux of five canonical Runge-Kutta
# methods, with algebraic verification of the first three Butcher order
# conditions and printout of the stability function R(z).
#
# Methods covered:
#   1. Forward Euler           (s=1, order 1)
#   2. Midpoint RK2            (s=2, order 2)
#   3. Heun's method (RK2)     (s=2, order 2)
#   4. Classical RK4           (s=4, order 4)
#   5. Runge-Kutta-Fehlberg 4(5) (s=6, order 5)
#
# Algebraic verification: each method must satisfy
#     ∑ b_i           = 1         (order ≥ 1)
#     ∑ b_i c_i       = 1/2       (order ≥ 2)
#     ∑ b_i c_i²      = 1/3       (order ≥ 3)
#     ∑_{i,j} b_i a_{ij} c_j = 1/6   (order ≥ 3, second condition)
#
# Stability function: for an explicit method,
#     R(z) = 1 + z · bᵀ · (I − z A)⁻¹ · 𝟙.
# For nilpotent A, (I − zA)⁻¹ is a finite polynomial in z, so R(z) is a
# polynomial of degree ≤ s.
#
# Usage:
#     julia --project=. companions/ch05/julia/butcher_tableau_zoo.jl

using LinearAlgebra
using Printf

# ---------------------------------------------------------------------------
# Method tableaux
# ---------------------------------------------------------------------------

struct Tableau
    name::String
    A::Matrix{Float64}
    b::Vector{Float64}
    c::Vector{Float64}
    expected_order::Int
end

function forward_euler()
    return Tableau("Forward Euler", reshape([0.0], 1, 1), [1.0], [0.0], 1)
end

function midpoint_rk2()
    A = [0.0 0.0; 0.5 0.0]
    b = [0.0, 1.0]
    c = [0.0, 0.5]
    return Tableau("Midpoint RK2", A, b, c, 2)
end

function heun_rk2()
    A = [0.0 0.0; 1.0 0.0]
    b = [0.5, 0.5]
    c = [0.0, 1.0]
    return Tableau("Heun's RK2", A, b, c, 2)
end

function classical_rk4()
    A = [0.0  0.0  0.0  0.0;
         0.5  0.0  0.0  0.0;
         0.0  0.5  0.0  0.0;
         0.0  0.0  1.0  0.0]
    b = [1/6, 1/3, 1/3, 1/6]
    c = [0.0, 0.5, 0.5, 1.0]
    return Tableau("Classical RK4", A, b, c, 4)
end

function rkf45()
    # Fehlberg 1969 (5th-order weights)
    A = [0.0           0.0          0.0          0.0           0.0       0.0;
         1/4           0.0          0.0          0.0           0.0       0.0;
         3/32          9/32         0.0          0.0           0.0       0.0;
         1932/2197    -7200/2197    7296/2197    0.0           0.0       0.0;
         439/216      -8.0          3680/513    -845/4104      0.0       0.0;
        -8/27          2.0         -3544/2565    1859/4104    -11/40     0.0]
    b = [16/135, 0.0, 6656/12825, 28561/56430, -9/50, 2/55]
    c = [0.0, 1/4, 3/8, 12/13, 1.0, 1/2]
    return Tableau("RKF 4(5)", A, b, c, 5)
end

# ---------------------------------------------------------------------------
# Order conditions
# ---------------------------------------------------------------------------

"""
    verify_order_conditions(tab) -> NamedTuple

Compute the first three Butcher order-condition residuals for tableau `tab`.
Returns the sums and the difference from each target value (1, 1/2, 1/3, 1/6).
"""
function verify_order_conditions(tab::Tableau)
    s = length(tab.b)
    sum1 = sum(tab.b)                                # target: 1
    sum2 = sum(tab.b .* tab.c)                       # target: 1/2
    sum3 = sum(tab.b .* tab.c.^2)                    # target: 1/3
    sum4 = sum(tab.b[i] * tab.A[i, j] * tab.c[j]     # target: 1/6
               for i in 1:s, j in 1:s)
    return (s = s, sum1 = sum1, sum2 = sum2, sum3 = sum3, sum4 = sum4)
end

# ---------------------------------------------------------------------------
# Stability function and Taylor coefficients
# ---------------------------------------------------------------------------

"""
    stability_taylor(tab; degree=8) -> Vector{Float64}

Return the Taylor coefficients of R(z) = 1 + z · bᵀ · (I − z A)⁻¹ · 𝟙 up to
the specified degree. For an explicit method, the resulting coefficients
match exp(z) through the method's order (this IS the algebraic content of
the order conditions, viewed from the stability-function side).
"""
function stability_taylor(tab::Tableau; degree::Int = 8)
    s = length(tab.b)
    A, b = tab.A, tab.b
    # (I − zA)⁻¹ = Σ_{k=0}^{s-1} z^k A^k (geometric series; truncates because
    # A is nilpotent for explicit methods).
    ones_vec = ones(s)
    Ak = Matrix{Float64}(I, s, s)
    coeffs = zeros(Float64, degree + 1)
    coeffs[1] = 1.0   # constant term
    for k in 1:degree
        # The coefficient of z^k in R(z) is b' A^{k-1} 1 for k ≥ 1.
        coeffs[k + 1] = dot(b, Ak * ones_vec)
        Ak = Ak * A
        # No additional rescaling: this is exactly the expansion above.
    end
    return coeffs
end

"""
    expected_exp_coeffs(degree) -> Vector{Float64}

Taylor coefficients of exp(z): 1, 1, 1/2, 1/6, 1/24, ... = 1/k!.
"""
function expected_exp_coeffs(degree::Int)
    return [1.0 / factorial(k) for k in 0:degree]
end

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

function tableau_to_string(tab::Tableau)
    s = length(tab.b)
    lines = String[]
    push!(lines, "$(tab.name) (s = $s, expected order $(tab.expected_order))")
    push!(lines, "-" ^ 60)
    # Print A (one row per stage), with c values on the left.
    for i in 1:s
        row = @sprintf("c_%d = %7.4f  |", i, tab.c[i])
        for j in 1:s
            row *= @sprintf(" %9.4f", tab.A[i, j])
        end
        push!(lines, row)
    end
    bstr = "             b  |"
    for j in 1:s
        bstr *= @sprintf(" %9.4f", tab.b[j])
    end
    push!(lines, bstr)
    return join(lines, "\n")
end

function main()
    methods = [forward_euler(), midpoint_rk2(), heun_rk2(), classical_rk4(), rkf45()]

    println("=" ^ 70)
    println("Chapter 5 Julia companion: Butcher tableau zoo")
    println("=" ^ 70)
    println()

    for tab in methods
        println(tableau_to_string(tab))
        oc = verify_order_conditions(tab)
        @printf("\nOrder condition residuals:\n")
        @printf("  ∑ b_i             = %8.5f   (target 1.00000)\n", oc.sum1)
        @printf("  ∑ b_i c_i         = %8.5f   (target 0.50000)\n", oc.sum2)
        @printf("  ∑ b_i c_i^2       = %8.5f   (target 0.33333)\n", oc.sum3)
        @printf("  ∑ b_i a_ij c_j    = %8.5f   (target 0.16667)\n", oc.sum4)

        # Stability function Taylor coefficients
        coeffs = stability_taylor(tab; degree = 6)
        expected = expected_exp_coeffs(6)
        println("\nStability function R(z) Taylor coefficients vs exp(z):")
        @printf("  k=0  R(0) = %8.5f   exp(0) = %8.5f\n", coeffs[1], expected[1])
        for k in 1:6
            match_marker = abs(coeffs[k + 1] - expected[k + 1]) < 1e-10 ? "✓ matches exp(z)" : "differs from exp(z)"
            @printf("  k=%d  R^(%d)/k!  = %8.5f   1/%d! = %8.5f   (%s)\n",
                    k, k, coeffs[k + 1], k, expected[k + 1], match_marker)
        end

        println()
        println("-" ^ 70)
        println()
    end

    println("Interpretation:")
    println("  A method of order p has R(z) matching exp(z) Taylor coefficients")
    println("  through degree p, and differs from order p+1 onward. Forward Euler")
    println("  matches through k=1 only (order 1); RK2 through k=2; RK4 through")
    println("  k=4; RKF 4(5) through k=5.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
