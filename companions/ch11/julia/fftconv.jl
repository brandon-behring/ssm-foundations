# fftconv.jl --- Chapter 11 Julia companion: Hyena's causal long convolution.
#
# A causal long convolution is a linear time-invariant operator (Chapter 8's
# convolutional view). The frequency-domain identity is the teaching object: a
# convolution is a *pointwise product* of spectra, and zero-padding to length 2L
# makes the cyclic transform compute the *linear* (causal) convolution — without
# it, late taps wrap around onto early outputs and causality breaks.
#
# Stdlib-only note: Julia's FFT lives in the external FFTW.jl, not the standard
# library, so this companion uses an explicit dense DFT matrix (LinearAlgebra
# only) to stay in the fast stdlib-only test loop. That makes the transform
# O(L^2) rather than FFTW's O(L log L) — the asymptotic win is demonstrated in the
# JAX/torch companions; here we pin the *identity* `dftconv == naive-Toeplitz` to
# < 1e-12 and the 2L-padding causality, in a third language.
#
# Contract (matching the JAX/torch companions):
#   u    :: Array{Float64,3}  (B, L, D)   input
#   k    :: Matrix{Float64}   (D, L)      per-channel causal filter, k[d, τ] = tap at lag τ-1
#   bias :: Vector{Float64}   (D,)        per-channel feedthrough
#   y[b,t,d] = sum_{s<=t} k[d, t-s+1] u[b,s,d] + bias[d] u[b,t,d]
#
# Port credit: mirrors companions/ch11/jax/fftconv.py, itself ported from
# post_transformers/experiments/jax/week11/hyena_lineage.py. Hyena: Poli et al.,
# arXiv:2302.10866.

using LinearAlgebra

"""
    dft_matrix(N) -> Matrix{ComplexF64}

The `N×N` DFT matrix `F[j+1,k+1] = exp(-2πi·jk/N)` (reducing `jk mod N` for
accuracy). `F * x` is the forward DFT; `conj(F)/N * X` the inverse.
"""
function dft_matrix(N::Int)
    F = Matrix{ComplexF64}(undef, N, N)
    @inbounds for j in 0:(N - 1), k in 0:(N - 1)
        F[j + 1, k + 1] = cis(-2π * ((j * k) % N) / N)
    end
    return F
end

function _check(u::Array{Float64,3}, k::Matrix{Float64}, bias::Vector{Float64})
    B, L, D = size(u)
    size(k) == (D, L) || throw(ArgumentError("k must be (D, L) = ($D, $L), got $(size(k))"))
    length(bias) == D || throw(ArgumentError("bias must be length D = $D, got $(length(bias))"))
    return B, L, D
end

"""
    dftconv(u, k, bias) -> Array{Float64,3}

Causal long convolution via the 2L-padded dense DFT. Equals [`causal_conv1d_naive`](@ref)
to machine precision (the `ch11:fftconv-causal` identity).
"""
function dftconv(u::Array{Float64,3}, k::Matrix{Float64}, bias::Vector{Float64})
    B, L, D = _check(u, k, bias)
    N = 2L
    F = dft_matrix(N)
    Finv = conj(F) / N
    y = zeros(Float64, B, L, D)
    @inbounds for d in 1:D
        kpad = zeros(ComplexF64, N)
        kpad[1:L] .= k[d, :]
        Kf = F * kpad
        for b in 1:B
            upad = zeros(ComplexF64, N)
            upad[1:L] .= u[b, :, d]
            conv = real(Finv * ((F * upad) .* Kf))
            y[b, :, d] .= conv[1:L] .+ bias[d] .* u[b, :, d]
        end
    end
    return y
end

"""
    causal_conv1d_naive(u, k, bias) -> Array{Float64,3}

O(L^2) explicit lower-triangular Toeplitz reference. Ground-truth oracle.
"""
function causal_conv1d_naive(u::Array{Float64,3}, k::Matrix{Float64}, bias::Vector{Float64})
    B, L, D = _check(u, k, bias)
    y = zeros(Float64, B, L, D)
    @inbounds for d in 1:D, b in 1:B, t in 1:L
        acc = 0.0
        for s in 1:t
            acc += k[d, t - s + 1] * u[b, s, d]  # lag (t-s), 1-based index t-s+1
        end
        y[b, t, d] = acc + bias[d] * u[b, t, d]
    end
    return y
end

"""
    cyclic_conv_unpadded(u, k, bias) -> Array{Float64,3}

Deliberately-wrong length-L (un-padded) cyclic DFT convolution. Wraps late taps
onto early outputs, so it is NOT causal and disagrees with the oracle —
demonstrating that the 2L padding is necessary.
"""
function cyclic_conv_unpadded(u::Array{Float64,3}, k::Matrix{Float64}, bias::Vector{Float64})
    B, L, D = _check(u, k, bias)
    F = dft_matrix(L)
    Finv = conj(F) / L
    y = zeros(Float64, B, L, D)
    @inbounds for d in 1:D
        Kf = F * ComplexF64.(k[d, :])
        for b in 1:B
            conv = real(Finv * ((F * ComplexF64.(u[b, :, d])) .* Kf))
            y[b, :, d] .= conv .+ bias[d] .* u[b, :, d]
        end
    end
    return y
end
