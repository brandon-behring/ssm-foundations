# xlstm_stabilization.jl --- Chapter 13 Julia companion (stdlib only).
#
# Cross-language check of §13.4: the mLSTM matrix-memory recurrence with
# exponential gates (naive, overflow-prone) and the log-domain max-state
# stabilizer. Mirrors companions/ch13/jax/xlstm.py; the load-bearing facts must
# agree with the JAX companion:
#   * P2 (stabilizer exactness): wherever the naive recurrence does not overflow,
#     the stabilized readout equals it to < 1e-12 (a change of variables, not an
#     approximation);
#   * rescaled gates f', i' lie in (0, 1] by construction of m_t as a running max;
#   * the naive recurrence overflows float64 once exp(log_i) exceeds ~1.8e308
#     (log_i >~ 709), while the stabilized one stays finite;
#   * single-pair recovery: storing (k, v) on a unit key and reading at q = k
#     returns exactly v = [0.3, -0.7, 1.1] for ANY log_i, including log_i = 800
#     where the naive recurrence overflows (the cross-language numeric anchor,
#     shared verbatim with the JAX companion's _REF_K_RAW / _REF_V).
#
# Port credit: greenfield from xLSTM (arXiv:2405.04517 §4); mirrors the JAX module.

using LinearAlgebra

"""
    log_sigmoid(x)

`log σ(x) = -softplus(-x) ≤ 0` (the forget log-gate), numerically stable.
"""
log_sigmoid(x::Real) = -(max(-x, 0.0) + log1p(exp(-abs(x))))

"""
    mlstm_naive(q, k, v, log_f, log_i)

mLSTM readouts with raw exponential gates `f = exp(log_f)`, `i = exp(log_i)`:
the matrix memory `C` and normalizer `n` accumulate without rescaling, so a large
`log_i` overflows float64 and the readout becomes `Inf`/`NaN`.

`q, k` are `(L, d_k)` (rows are timesteps), `v` is `(L, d_v)`, `log_f, log_i` are
length-`L`. Returns the `(L, d_v)` readouts.
"""
function mlstm_naive(q::AbstractMatrix, k::AbstractMatrix, v::AbstractMatrix,
                     log_f::AbstractVector, log_i::AbstractVector)
    L, d_k = size(q)
    d_v = size(v, 2)
    (size(k) == (L, d_k) && size(v, 1) == L) ||
        throw(ArgumentError("q, k must be (L, d_k); v must be (L, d_v)"))
    (length(log_f) == L && length(log_i) == L) ||
        throw(ArgumentError("log_f, log_i must have length L = $L"))
    cell = zeros(d_v, d_k)
    nrm = zeros(d_k)
    H = zeros(L, d_v)
    for t in 1:L
        f = exp(log_f[t])
        i = exp(log_i[t])
        cell = f .* cell .+ i .* (v[t, :] * k[t, :]')
        nrm = f .* nrm .+ i .* k[t, :]
        denom = max(abs(dot(nrm, q[t, :])), 1.0)
        H[t, :] = (cell * q[t, :]) ./ denom
    end
    return H
end

"""
    mlstm_stabilized(q, k, v, log_f, log_i)

mLSTM readouts via the log-domain max-state stabilizer
`m_t = max(log_f_t + m_{t-1}, log_i_t)`, `m_0 = -Inf`. Both rescaled gates lie in
`(0, 1]`, so nothing overflows; the readout floor `max(·, 1)` becomes
`max(·, exp(-m_t))` — the only change (P2). Returns `(H, m_trajectory)`.
"""
function mlstm_stabilized(q::AbstractMatrix, k::AbstractMatrix, v::AbstractMatrix,
                          log_f::AbstractVector, log_i::AbstractVector)
    L, d_k = size(q)
    d_v = size(v, 2)
    (size(k) == (L, d_k) && size(v, 1) == L) ||
        throw(ArgumentError("q, k must be (L, d_k); v must be (L, d_v)"))
    (length(log_f) == L && length(log_i) == L) ||
        throw(ArgumentError("log_f, log_i must have length L = $L"))
    cell = zeros(d_v, d_k)
    nrm = zeros(d_k)
    m = -Inf
    H = zeros(L, d_v)
    mtraj = zeros(L)
    for t in 1:L
        m_new = max(log_f[t] + m, log_i[t])
        f_p = exp(log_f[t] + m - m_new)  # in (0, 1]
        i_p = exp(log_i[t] - m_new)      # in (0, 1]
        cell = f_p .* cell .+ i_p .* (v[t, :] * k[t, :]')
        nrm = f_p .* nrm .+ i_p .* k[t, :]
        denom = max(abs(dot(nrm, q[t, :])), exp(-m_new))
        H[t, :] = (cell * q[t, :]) ./ denom
        mtraj[t] = m_new
        m = m_new
    end
    return H, mtraj
end

# The fixed reference pair, shared verbatim with the JAX companion
# (_REF_K_RAW / _REF_V) so `single_pair_recovery` is a cross-language anchor.
const REF_K_RAW = [1.0, 2.0, -1.0, 0.5]
const REF_V = [0.3, -0.7, 1.1]

"""
    single_pair_recovery(log_i_value)

Store one pair on the unit reference key with input log-gate `log_i_value`, read
at `q = k`. A single write makes `C_1 = i'_1 v k'`, `n_1 = i'_1 k` with `i'_1 = 1`,
so the read is `v ||k||^2 / max(||k||^2, exp(-m_1)) = v` — exactly the stored value
for ANY `log_i`, including 800 where the naive recurrence overflows. Returns
`(max_abs_error_vs_REF_V, readout)`.
"""
function single_pair_recovery(log_i_value::Real)
    k = REF_K_RAW / norm(REF_K_RAW)
    v = REF_V
    q = reshape(k, 1, :)
    H, _ = mlstm_stabilized(q, reshape(k, 1, :), reshape(v, 1, :), [0.0], [log_i_value])
    readout = H[1, :]
    return maximum(abs.(readout .- v)), readout
end
