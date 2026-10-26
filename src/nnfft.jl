# NNFFT (type 3): f̂_j = Σ_l f_l exp(−2πi k_j·y_l) for arbitrary real nodes y_l (sources, D × N) and
# k_j (targets, D × J). The algorithm of Elbel and Steidl (1998) and Potts, Steidl and Tasche (2001),
# in the form of FINUFFT's type 3 (Barnett, Magland and af Klinteberg, SISC 41, 2019, Sec. 3.3):
#
# 1. Both node sets are centred, y = y₀ + y′, k = k₀ + k′, with |y′_d| ≤ Y_d, |k′_d| ≤ K_d; then
#    f̂_j = exp(−2πi k_j·y₀) Σ_l (f_l exp(−2πi k₀·y′_l)) exp(−2πi k′_j·y′_l).
# 2. The sources are spread with an outer window φ (width W₁ in grid units) onto a grid of n points
#    of spacing h, b_m = Σ_l f′_l φ(m − y′_l/h), m = −n÷2 … n − n÷2 − 1 (the adjoint resampling of an
#    NFFT core whose grid is exactly n).
# 3. An NFFT with modes m evaluates Σ_m b_m exp(−2πi m x_j) at x_j = h k′_j; by the Poisson summation
#    formula this is φ̂(h k′_j) Σ_l f′_l exp(−2πi k′_j·y′_l) up to aliasing terms φ̂(h k′_j + r), r ≠ 0.
# 4. Division by φ̂(h k′_j) per dimension.
#
# h = 1/(2σ₁K) keeps |h k′| ≤ 1/(2σ₁), where the outer window has aliasing error ε₁ (its oversampling
# factor σ₁); n ≥ 2Y/h + W₁ + 2 keeps every support inside the grid, so the spreading never wraps. The
# error is about ε₁ + A·ε₂, ε₂ the error of the inner NFFT and A = φ̂(0)/φ̂(1/(2σ₁)) the amplification
# of the division (the dynamic range of FINUFFT's Remark 3).

"""
    NNFFTPlan{T,D}

NNFFT plan of NFFT.jl (non-equidistant nodes in both domains), created by `plan_nnfft(k, y; kwargs...)`
for targets `k` (D × J) and sources `y` (D × N), arbitrary real coordinates:

    f̂_j = Σ_l f_l exp(−2πi k_j·y_l)

The cost grows with the product of the extents of `k` and `y` (the size of the inner NFFT is about
4σ²·Y·K per dimension), not with their numbers. Generic options `reltol`, `eltype`, `storage`,
`nthreads` and the options of [`NFFTPlan`](@ref) (`window`, `W`, `σ` for the inner NFFT).
"""
mutable struct NNFFTPlan{T,D} <: AbstractNNFFTPlan{T,1,1}
  N::Int                       # number of sources
  J::Int                       # number of targets
  k::AbstractMatrix
  y::AbstractMatrix
  outer::KernelEntry           # window of the spreading
  n::NTuple{D,Int}             # grid of the spreading = modes of the inner NFFT
  accuracy::Float64
  options::GenericOptions{T}
  core::Any                    # spreading: NFFT core on a grid of exactly n points
  inner::Any                   # NFFT plan of size n at the nodes h k′
  pre::AbstractVector          # exp(−2πi k₀·y′_l)
  post::AbstractVector         # exp(−2πi k_j·y₀) / Π_d φ̂(h_d k′_dj)
  g::AbstractArray             # spread grid (n...)
  fs::AbstractVector           # sources times pre
  fj::AbstractVector           # targets times conj(post)
end

const NNFFT_OPTIONS = (:window, :W, :σ, :fft)

function AbstractNFFTs.plan_nnfft(b::NFFTBackend, k::AbstractMatrix, y::AbstractMatrix; kwargs...)
  D = size(k, 1)
  size(y, 1) == D || throw(ArgumentError("targets have $D dimensions, sources $(size(y, 1))"))
  g, rest = generic_options(k, ntuple(_ -> 1, D); kwargs...)
  g.dims == 1:D || throw(ArgumentError("an NNFFT has no batch dimensions"))
  check_options(b, rest, NNFFT_OPTIONS)
  NNFFTPlan(k, y, g; rest...)
end

NNFFTPlan(k::AbstractVecOrMat, y::AbstractVecOrMat; kwargs...) = plan_nnfft(NFFTBackend(), k, y; kwargs...)

function NNFFTPlan(k::AbstractMatrix, y::AbstractMatrix, g::GenericOptions{T}; window=:auto, W=nothing, σ=nothing,
                   fft=FFTWBackend()) where {T}
  kc = Array{Float64}(k); yc = Array{Float64}(y)
  all(isfinite, kc) && all(isfinite, yc) || throw(ArgumentError("nodes must be finite"))
  D = size(kc, 1); J = size(kc, 2); N = size(yc, 2)
  center(x) = isempty(x) ? zeros(D) : [(minimum(x[d, :]) + maximum(x[d, :])) / 2 for d in 1:D]
  extent(x, c) = isempty(x) ? zeros(D) : [maximum(abs, x[d, :] .- c[d]) for d in 1:D]
  y0 = center(yc); k0 = center(kc)
  Y = extent(yc, y0); K = extent(kc, k0)
  # outer window: half the error budget at the grid the spreading will have
  nest = ntuple(d -> ceil(Int, 8Y[d] * K[d]) + 16, D)
  outer, e1 = choose_kernel(g.reltol / 2, nest, N + J, T)
  σ1 = outer.σ; W1 = outer.W
  h = [K[d] > 0 ? 1 / (2σ1 * K[d]) : 1.0 for d in 1:D]
  n = ntuple(d -> smooth_size(ceil(Int, 2Y[d] / h[d]) + W1 + 2), D)
  # the outer grid as an NFFT core: source y′/h sits at grid position y′/h + n÷2
  ys = Matrix{Float64}(undef, D, N)
  for l in 1:N, d in 1:D
    x = (yc[d, l] - y0[d]) / (h[d] * n[d]) + (n[d] ÷ 2) / n[d]
    ys[d, l] = x >= 0.5 ? x - 1 : x
  end
  φ = window_function(outer.window, W1, outer.β, outer.e0)
  core = g.storage <: Array ?
    NFFTCore(ys, n, n, W1, φ, T, FFTWBackend(), nothing, g.nthreads) :
    device_core(g.storage, ys, n, n, W1, φ, T, g.nthreads, 1, 1)
  # inner NFFT at h k′ in Float64 (its grid positions are computed from these)
  xs = (kc .- k0) .* h
  inner = plan_nfft(NFFTBackend(), xs, n; reltol=g.reltol / (2outer.amp), eltype=T, storage=g.storage,
                    nthreads=g.nthreads, window, W, σ, fft)
  # phases and the division by φ̂ in Float64
  x, w = gausslegendre(2W1 + 20)
  z = x .* (W1 / 2); wz = w .* (W1 / 2) .* φ.(z)
  φ̂(ξ) = sum(wz[q] * cospi(2ξ * z[q]) for q in eachindex(z))
  pre = [cispi(-2 * sum(k0[d] * (yc[d, l] - y0[d]) for d in 1:D; init=0.0)) for l in 1:N]
  post = Vector{ComplexF64}(undef, J)
  Threads.@threads for j in 1:J
    post[j] = cispi(-2 * sum(kc[d, j] * y0[d] for d in 1:D; init=0.0)) / prod(φ̂(xs[d, j]) for d in 1:D)
  end
  S = g.storage
  store(v) = to_storage(S, Complex{T}.(v))
  acc = e1 + outer.amp * accuracy(inner)
  NNFFTPlan{T,D}(N, J, to_storage(S, k, T), to_storage(S, y, T), outer, n, acc, g, core, inner, store(pre), store(post),
                 to_storage(S, zeros(Complex{T}, n)), to_storage(S, zeros(Complex{T}, N)),
                 to_storage(S, zeros(Complex{T}, J)))
end

AbstractNFFTs.size_in(p::NNFFTPlan) = (p.N,)
AbstractNFFTs.size_out(p::NNFFTPlan) = (p.J,)
AbstractNFFTs.nodes(p::NNFFTPlan) = p.k
AbstractNFFTs.accuracy(p::NNFFTPlan) = p.accuracy

function AbstractNFFTs.nodes!(p::NNFFTPlan{T}, k::AbstractMatrix) where {T}
  q = NNFFTPlan(k, p.y, p.options; parameters(p.inner)...)
  for f in fieldnames(NNFFTPlan)
    setfield!(p, f, getfield(q, f))
  end
  p
end

Base.copy(p::NNFFTPlan) = NNFFTPlan(p.k, p.y, p.options; parameters(p.inner)...)

function Base.show(io::IO, p::NNFFTPlan{T}) where {T}
  print(io, "NNFFTPlan{$T} from $(p.N) sources to $(p.J) targets: inner NFFT of size $(p.n), ",
        "spreading with $(nameof(typeof(p.outer.window))) W = $(p.outer.W), σ = $(p.outer.σ), predicted error $(p.accuracy)")
end

function LinearAlgebra.mul!(fHat::AbstractVector, p::NNFFTPlan, f::AbstractVector)
  length(f) == p.N || throw(DimensionMismatch("input has length $(length(f)), the plan expects $(p.N)"))
  length(fHat) == p.J || throw(DimensionMismatch("output has length $(length(fHat)), the plan expects $(p.J)"))
  p.fs .= f .* p.pre
  resampling_adjoint_stage!(p.core, p.g, p.fs)
  mul!(fHat, p.inner, p.g)
  fHat .*= p.post
end

function LinearAlgebra.mul!(f::AbstractVector, pl::Adjoint{Complex{T},<:NNFFTPlan{T}}, fHat::AbstractVector) where {T}
  p = pl.parent
  length(f) == p.N || throw(DimensionMismatch("output has length $(length(f)), the plan expects $(p.N)"))
  length(fHat) == p.J || throw(DimensionMismatch("input has length $(length(fHat)), the plan expects $(p.J)"))
  p.fj .= fHat .* conj.(p.post)
  mul!(p.g, adjoint(p.inner), p.fj)
  resampling_stage!(p.core, f, p.g)
  f .*= conj.(p.pre)
end
