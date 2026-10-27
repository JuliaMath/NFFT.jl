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
    NNFFTPlan{T,D,M}

NNFFT plan of NFFT.jl (non-equidistant nodes in both domains), created by
`plan_nnfft(k, y[, N]; kwargs...)` for targets `k` (D × J) and sources `y` (D × N), arbitrary real
coordinates:

    f̂_j = Σ_l f_l exp(−2πi k_j·y_l)

The cost grows with the product of the extents of `k` and `y` (the size of the inner NFFT is about
4σ²·Y·K per dimension), not with their numbers. With an array size `N` (M dimensions), `dims`
selects the dimension of the sources; the transform maps it to the targets and passes the other
dimensions through, one slice after the other. Generic options `reltol`, `eltype`, `storage`,
`nthreads`; restrictions of the inner NFFT `window` (`:auto`, `KaiserBessel()` or
`ExpSemicircle()`), `W` and `σ`, and its FFT backend `fft`, as for [`NFFTPlan`](@ref).
"""
mutable struct NNFFTPlan{T,D,M} <: AbstractNNFFTPlan{T,M,M}
  N::Int                       # number of sources
  J::Int                       # number of targets
  Nin::NTuple{M,Int}           # input size: the sources along `dim`, batch dimensions around it
  NOut::NTuple{M,Int}          # output size: the targets along `dim`
  dim::Int
  k::AbstractMatrix
  y::AbstractMatrix
  ysrc::AbstractMatrix         # sources in full precision on the CPU, for nodes!
  outer::KernelEntry           # window of the spreading
  n::NTuple{D,Int}             # grid of the spreading = modes of the inner NFFT
  accuracy::Float64
  options::GenericOptions{T}
  core::Any                    # spreading: NFFT core on a grid of exactly n points
  inner::Any                   # NFFT plan of size n at the nodes h k′
  pre::AbstractVector          # exp(−2πi k₀·y′_l)
  post::AbstractVector         # exp(−2πi k_j·y₀) / Π_d φ̂(h_d k′_dj)
  g::AbstractArray             # spread grid (n...); empty on the CPU, where the core's grid is used
  fs::AbstractVector           # sources times pre
  fj::AbstractVector           # targets times conj(post)
end

const NNFFT_OPTIONS = (:window, :W, :σ, :fft)

# `N` is the size of the data array and `dims` the dimension of the sources in it; the other
# dimensions are batch dimensions. The nodes have D rows independently of it, so `dims` is not
# passed on to `generic_options`.
function AbstractNFFTs.plan_nnfft(b::NFFTBackend, k::AbstractMatrix, y::AbstractMatrix, N::Dims=(size(y, 2),);
                                  dims=nothing, kwargs...)
  D = size(k, 1)
  size(y, 1) == D || throw(ArgumentError("targets have $D dimensions, sources $(size(y, 1))"))
  dim = source_dim(dims, N)
  N[dim] == size(y, 2) ||
    throw(DimensionMismatch("dimension $dim of the size $N has $(N[dim]) entries, there are $(size(y, 2)) sources"))
  g, rest = generic_options(k, ntuple(_ -> 1, D); kwargs...)
  g = GenericOptions(g.reltol, dim:dim, g.eltype, g.storage, g.nthreads)
  check_options(b, rest, NNFFT_OPTIONS)
  NNFFTPlan(k, y, g; N, rest...)
end

function source_dim(dims, N)
  d = dims === nothing && length(N) == 1 ? 1 : dims isa Integer ? Int(dims) :
      dims isa AbstractUnitRange && length(dims) == 1 ? first(dims) :
      throw(ArgumentError("dims must be the dimension of the sources (an integer), got $(repr(dims))"))
  1 <= d <= length(N) || throw(ArgumentError("dims = $d is not a dimension of the size $N"))
  d
end

NNFFTPlan(k::AbstractVecOrMat, y::AbstractVecOrMat, N::Dims...; kwargs...) = plan_nnfft(NFFTBackend(), k, y, N...; kwargs...)

function NNFFTPlan(k::AbstractMatrix, y::AbstractMatrix, g::GenericOptions{T}; N=(size(y, 2),), window=:auto,
                   W=nothing, σ=nothing, fft=nothing) where {T}
  cpu_options(g.storage, fft, nothing)
  kc = Array{Float64}(k); yc = Array{Float64}(y)
  all(isfinite, kc) && all(isfinite, yc) || throw(ArgumentError("nodes must be finite"))
  nnfft_plan(Val(size(kc, 1)), k, y, kc, yc, g, N; window, W, σ, fft)
end

function nnfft_plan(::Val{D}, k, y, kc, yc, g::GenericOptions{T}, Nin::NTuple{M,Int}; window, W, σ, fft) where {D,T,M}
  J = size(kc, 2); N = size(yc, 2)
  y0 = ntuple(d -> N == 0 ? 0.0 : (minimum(view(yc, d, :)) + maximum(view(yc, d, :))) / 2, Val(D))
  k0 = ntuple(d -> J == 0 ? 0.0 : (minimum(view(kc, d, :)) + maximum(view(kc, d, :))) / 2, Val(D))
  Y = ntuple(d -> N == 0 ? 0.0 : maximum(x -> abs(x - y0[d]), view(yc, d, :)), Val(D))
  K = ntuple(d -> J == 0 ? 0.0 : maximum(x -> abs(x - k0[d]), view(kc, d, :)), Val(D))
  outer, inner_entry, n, acc = choose_nnfft(g.reltol, Y, K, N, J, T; window, W, σ)
  σ1 = outer.σ; W1 = outer.W
  h = map(x -> x > 0 ? 1 / (2σ1 * x) : 1.0, K)
  # the outer grid as a resampling core: source y′/h sits at grid position y′/h + n÷2
  ys = Matrix{Float64}(undef, D, N)
  @maybe_threads g.nthreads for l in 1:N
    for d in 1:D
      x = (yc[d, l] - y0[d]) / (h[d] * n[d]) + (n[d] ÷ 2) / n[d]
      ys[d, l] = x >= 0.5 ? x - 1 : x
    end
  end
  φ = window_function(outer.window, W1, outer.β, outer.e0)
  core = g.storage <: Array ?
    NFFTCore(ys, n, n, W1, φ, T, nothing, nothing, g.nthreads) :
    device_core(g.storage, ys, n, n, W1, φ, T, g.nthreads, 1, 1; fft=false)
  # inner NFFT at h k′ in Float64 (its grid positions are computed from these)
  xs = Matrix{Float64}(undef, D, J)
  @maybe_threads g.nthreads for j in 1:J
    for d in 1:D
      xs[d, j] = (kc[d, j] - k0[d]) * h[d]
    end
  end
  e2 = inner_entry
  # the kernel is fixed by the choice above; below the reachable error the tolerance only must stay positive
  inner = plan_nfft(NFFTBackend(), xs, n; reltol=max((g.reltol - acc.outer) / outer.amp, eps(Float64)), eltype=T, storage=g.storage,
                    nthreads=g.nthreads, window=e2.window, W=e2.W, σ=e2.σ, fft)
  # phases and the division by φ̂ in Float64; |h k′| ≤ 1/(2σ₁)
  ft = WindowFT(φ, W1, 1 / (2σ1))
  pre = Vector{ComplexF64}(undef, N)
  @maybe_threads g.nthreads for l in 1:N
    pre[l] = cispi(-2 * sum(ntuple(d -> k0[d] * (yc[d, l] - y0[d]), Val(D))))
  end
  post = Vector{ComplexF64}(undef, J)
  @maybe_threads g.nthreads for j in 1:J
    post[j] = cispi(-2 * sum(ntuple(d -> kc[d, j] * y0[d], Val(D)))) / prod(ntuple(d -> ft(abs(xs[d, j])), Val(D)))
  end
  S = g.storage
  store(v) = to_storage(S, Complex{T}.(v))
  dim = first(g.dims)
  NNFFTPlan{T,D,M}(N, J, Nin, Base.setindex(Nin, J, dim), dim, to_storage(S, k, T), to_storage(S, y, T), S <: Array ? y : yc, outer, n, acc.outer + outer.amp * accuracy(inner),
                 g, core, inner, store(pre), store(post), to_storage(S, zeros(Complex{T}, S <: Array ? map(zero, n) : n)),
                 to_storage(S, zeros(Complex{T}, N)), to_storage(S, zeros(Complex{T}, J)))
end

# Outer window e₁ (spreading) and inner NFFT kernel e₂ at the lowest predicted cost: e₁ with error
# ε₁ fixes the grid n ≈ 4σ₁·Y·K and leaves (reltol − ε₁)/A₁ for the inner NFFT. Cost:
# the spreading term of the cost model for the N sources plus the cost of the inner NFFT (both
# directions). `window`, `W`, `σ` restrict the inner kernel.
function choose_nnfft(reltol, Y::NTuple{D}, K::NTuple{D}, N, J, ::Type{T}; window=:auto, W=nothing, σ=nothing) where {D,T}
  check_window_options(W, σ)
  window === :auto || window isa Union{KaiserBessel,ExpSemicircle} ||
    throw(ArgumentError("window must be :auto, KaiserBessel() or ExpSemicircle(), got $window"))
  a, a1, b, c = COST_COEF[min(D, 3)]
  grid(e1) = ntuple(d -> smooth_size(ceil(Int, 4 * e1.σ * Y[d] * K[d]) + e1.W + 2), Val(D))
  inner = filter(KERNEL_TABLE) do e
    (window === :auto || e.window == window) && (W === nothing || e.W == W)
  end
  if σ !== nothing
    σtab = argmin(s -> abs(s - σ), unique(e.σ for e in inner))
    inner = filter(e -> e.σ == σtab, inner)
  end
  isempty(inner) && throw(ArgumentError("no table entry for window = $window, W = $W, σ = $σ"))
  # Only kernels on the Pareto front matter: an inner kernel is dropped if another one is at least as
  # accurate with no larger W and σ, an outer one if another one also has no larger amplification.
  ε(e) = window_error(e, D) + 10 * eps(T) * e.amp
  inner = pareto_front(inner, e -> (ε(e), e.W, e.σ))
  outer = pareto_front(KERNEL_TABLE, e -> (ε(e), e.W, e.σ, e.amp))
  # inner error without the floor (which depends on n), and the n-independent cost per kernel
  ε2 = [ε(e) for e in inner]
  c2 = [a * J * e.W^D + a1 * J * e.W for e in inner]
  σ2 = [e.σ^D for e in inner]
  # target: reltol, or 1.5 times the smallest predicted error if reltol is below it
  fl(e1) = error_floor(grid(e1), T)
  emin = minimum(e1 -> ε(e1) + fl(e1) + e1.amp * (minimum(ε2) + fl(e1)), outer)
  tol = max(reltol, 1.5emin)
  best = nothing; bestcost = Inf
  for e1 in outer
    ε(e1) <= tol || continue
    n = grid(e1)
    f = error_floor(n, T)
    ε1 = ε(e1) + f
    ε1 <= tol || continue
    tol2 = (tol - ε1) / e1.amp - f
    c1 = a * N * e1.W^D + a1 * N * e1.W + c
    Mn = prod(n)
    for i in eachindex(inner)
      ε2[i] <= tol2 || continue
      M = σ2[i] * Mn
      cost = c1 + c2[i] + b * M * log2(M)
      cost < bestcost && ((best, bestcost) = ((e1, inner[i], n, ε1), cost))
    end
  end
  e1, e2, n, ε1 = best
  e1, e2, n, (outer=ε1,)
end

# entries whose key tuple is not dominated (all components ≥, one >) by another entry's
function pareto_front(entries, key)
  ks = map(key, entries)
  order = sortperm(ks)
  front = Int[]
  for i in order
    any(j -> all(map(<=, ks[j], ks[i])), front) || push!(front, i)
  end
  entries[front]
end

AbstractNFFTs.size_in(p::NNFFTPlan) = p.Nin
AbstractNFFTs.size_out(p::NNFFTPlan) = p.NOut
AbstractNFFTs.nodes(p::NNFFTPlan) = p.k
AbstractNFFTs.accuracy(p::NNFFTPlan) = p.accuracy

function AbstractNFFTs.nodes!(p::NNFFTPlan{T,D}, k::AbstractMatrix) where {T,D}
  size(k, 1) == D || throw(DimensionMismatch("targets must have $D rows"))
  q = NNFFTPlan(k, p.ysrc, p.options; N=p.Nin, parameters(p.inner)...)
  for f in fieldnames(NNFFTPlan)
    setfield!(p, f, getfield(q, f))
  end
  p
end

# A copy shares nodes, phases and correction factors and gets its own cores and buffers.
Base.copy(p::NNFFTPlan{T,D,M}) where {T,D,M} =
  NNFFTPlan{T,D,M}(p.N, p.J, p.Nin, p.NOut, p.dim, p.k, p.y, p.ysrc, p.outer, p.n, p.accuracy, p.options, copy(p.core), copy(p.inner),
                 p.pre, p.post, similar(p.g), similar(p.fs), similar(p.fj))

function Base.show(io::IO, p::NNFFTPlan{T,D,M}) where {T,D,M}
  print(io, "NNFFTPlan{$T} from $(p.N) sources to $(p.J) targets",
        M > 1 ? " along dims $(p.dim) of an array of size $(p.Nin)" : "", ": inner NFFT of size $(p.n), ",
        "spreading with $(nameof(typeof(p.outer.window))) W = $(p.outer.W), σ = $(p.outer.σ), predicted error $(p.accuracy)")
end

# Batch dimensions: one slice after the other through the buffers of the plan.
function LinearAlgebra.mul!(fHat::AbstractArray, p::NNFFTPlan, f::AbstractArray)
  check_size(f, p.Nin, "input"); check_size(fHat, p.NOut, "output")
  L1, L3 = batch_sizes(p.Nin, p.dim:p.dim)
  L1 * L3 == 1 && (nnfft!(vec(fHat), p, vec(f)); return fHat)
  fr = reshape(f, L1, p.N, L3); gr = reshape(fHat, L1, p.J, L3)
  for b in 1:L3, a in 1:L1
    view(gr, a, :, b) .= nnfft!(p.fj, p, view(fr, a, :, b))
  end
  fHat
end

function nnfft!(fHat, p, f)
  p.fs .= f .* p.pre
  mul!(fHat, p.inner, spread!(p.core, p.g, p.fs))
  fHat .*= p.post
end

# spread the sources onto the grid; on the CPU the inner NFFT reads the grid of the core in place
spread!(c::NFFTCore, g, fs) = resample_adjoint_interior!(c, fs)
spread!(c, g, fs) = (resampling_adjoint_stage!(c, g, fs); g)

function LinearAlgebra.mul!(f::AbstractArray, pl::Adjoint{Complex{T},<:NNFFTPlan{T}}, fHat::AbstractArray) where {T}
  p = pl.parent
  check_size(f, p.Nin, "output"); check_size(fHat, p.NOut, "input")
  L1, L3 = batch_sizes(p.Nin, p.dim:p.dim)
  L1 * L3 == 1 && (nnfft_adjoint!(vec(f), p, vec(fHat)); return f)
  fr = reshape(f, L1, p.N, L3); gr = reshape(fHat, L1, p.J, L3)
  for b in 1:L3, a in 1:L1
    view(fr, a, :, b) .= nnfft_adjoint!(p.fs, p, view(gr, a, :, b))
  end
  f
end

function nnfft_adjoint!(f, p, fHat)
  p.fj .= fHat .* conj.(p.post)
  interpolate!(p.core, f, p.g, adjoint(p.inner), p.fj)
  f .*= conj.(p.pre)
end

# the adjoint of the inner NFFT and the resampling at the sources (on the CPU in place, as `spread!`)
function interpolate!(c::NFFTCore, f, g, inner, fj)
  mul!(interior(c.gp, halo(c.W)), inner, fj)
  resample_interior!(c, f)
end
interpolate!(c, f, g, inner, fj) = (mul!(g, inner, fj); resampling_stage!(c, f, g); f)
