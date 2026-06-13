# symplectic_crosscheck.jl --- Chapter 17 Julia companion (stdlib only).
#
# An independent-language cross-check of the §17.2 C1 atlas cell: the energy conservation of three
# integrators on the harmonic-oscillator SSM mode. Mirrors companions/ch17/jax/c1_integration.py
# and is the C1 atlas cell in the pilot's own language (the C1 symplectic_atlas is Julia; echoes
# ch10's Julia-discretization decision). The load-bearing facts must agree with the JAX companion:
#   * the exact-exponential transition of the imaginary mode conserves energy (band ~ 0);
#   * Störmer-Verlet (symplectic) has zero secular drift but a bounded oscillation band that
#     matches the JAX value;
#   * RK4 (non-symplectic) carries a secular drift whose endpoint value matches the JAX anchor
#     (the cross-language numeric anchor, computed from identical deterministic arithmetic).
#
# Port credit: greenfield; the integrators mirror companions/ch06/jax/symplectic_demo.py and the
# exact-exponential mode mirrors companions/ch10/jax/complex_state.py.

const TWO_PI = 2.0 * pi

n_steps(dt::Real, periods::Real) =
    (dt <= 0 || periods <= 0) ? throw(ArgumentError("dt and periods must be > 0")) :
    round(Int, periods * TWO_PI / dt)

"""Störmer-Verlet step for the harmonic oscillator (T'(p)=p, V'(q)=q); symplectic, 2nd order."""
function verlet_step(q::Float64, p::Float64, dt::Float64)
    p_half = p - 0.5 * dt * q
    q_next = q + dt * p_half
    p_next = p_half - 0.5 * dt * q_next
    return q_next, p_next
end

"""Classical RK4 step on ẋ = (p, -q) — not symplectic."""
function rk4_step(q::Float64, p::Float64, dt::Float64)
    f(qq, pp) = (pp, -qq)
    k1q, k1p = f(q, p)
    k2q, k2p = f(q + 0.5dt * k1q, p + 0.5dt * k1p)
    k3q, k3p = f(q + 0.5dt * k2q, p + 0.5dt * k2p)
    k4q, k4p = f(q + dt * k3q, p + dt * k3p)
    q_next = q + (dt / 6.0) * (k1q + 2k2q + 2k3q + k4q)
    p_next = p + (dt / 6.0) * (k1p + 2k2p + 2k3p + k4p)
    return q_next, p_next
end

"""Energy trajectory ``H_k = 0.5(q^2 + p^2)`` (length n_steps+1) under a stepper."""
function energy_trajectory(stepper, q0::Float64, p0::Float64, dt::Float64, n::Int)
    q, p = q0, p0
    es = Vector{Float64}(undef, n + 1)
    es[1] = 0.5 * (q^2 + p^2)
    for k in 1:n
        q, p = stepper(q, p, dt)
        es[k+1] = 0.5 * (q^2 + p^2)
    end
    return es
end

"""Exact-exponential complex mode ``z_k = e^{-i dt k} z_0``; energy ``0.5|z|^2`` (the diagonal SSM)."""
function exact_exp_energy(q0::Float64, p0::Float64, dt::Float64, n::Int)
    alpha = exp(-im * dt)
    z = complex(q0, p0)
    es = Vector{Float64}(undef, n + 1)
    es[1] = 0.5 * abs2(z)
    for k in 1:n
        z = alpha * z
        es[k+1] = 0.5 * abs2(z)
    end
    return es
end

energy_band(es::Vector{Float64}) = maximum(es) - minimum(es)

endpoint_drift_per_period(es::Vector{Float64}, dt::Float64) =
    (es[end] - es[1]) / ((length(es) - 1) * dt / TWO_PI)

"""Least-squares secular slope of energy vs period (robust to the bounded oscillation)."""
function secular_slope_per_period(es::Vector{Float64}, dt::Float64)
    n = length(es)
    x = collect(0:n-1) .* (dt / TWO_PI)
    xbar = sum(x) / n
    ybar = sum(es) / n
    return sum((x .- xbar) .* (es .- ybar)) / sum((x .- xbar) .^ 2)
end
