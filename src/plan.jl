# The engine of one transform over the D NFFT dimensions (batch dimensions are handled by the
# plan). Its type depends on the kernel (width W is a compile-time constant of the hot loops); the
# plan holds it in an untyped field, so plan types stay inferable and every transform crosses one
# function barrier into the specialised loops.
struct NFFTCore{D,K<:SymKernel,T,Tk}
  N::NTuple{D,Int}
  Ñ::NTuple{D,Int}
  W::Int
  B::Int
  kernel::K
  invphi::NTuple{D,Vector{T}}
  binned::Binned{D,Tk}
  gp::Array{Complex{T},D}           # padded grid, Ñ .+ W
  fwd::PassFFT{D}
  bwd::PassFFT{D}
  locks::TileLocks{D}
  nthreads::Int
end

function NFFTCore(k::AbstractMatrix, N::NTuple{D,Int}, Ñ::NTuple{D,Int}, W, φ, ::Type{T}, fft, B, nthreads) where {D,T}
  K = SymKernel(PolyKernel(φ, W, W + 3; T))
  invphi = ntuple(d -> T.(1 ./ kernel_ft(φ, W, N[d], Ñ[d]; nthreads)), D)
  B = B === nothing ? default_tile(D, W) : B
  h = halo(W)
  gp = Array{Complex{T}}(undef, Ñ .+ W)         # zeroed by every transform
  g = interior(gp, h)
  NFFTCore(N, Ñ, W, B, K, invphi, Binned(k, Ñ, B, nthreads), gp,
           PassFFT(fft, g, N; dir=:fwd, nthreads), PassFFT(fft, g, N; dir=:bwd, nthreads),
           TileLocks(size(gp), B), nthreads)
end

function trafo_core!(out, c::NFFTCore, f)
  h = halo(c.W); g = interior(c.gp, h)
  threaded_fill!(c.gp, 0, c.nthreads)
  resampling_correction_core!(g, f, c.invphi, c.N, c.Ñ, c.nthreads)
  apply!(c.fwd, g)
  fill_halo!(c.gp, h, c.nthreads)
  resample!(out, c.gp, c.kernel, c.binned, c.Ñ, c.B, c.nthreads)
  out
end

function adjoint_core!(f, c::NFFTCore, vals)
  h = halo(c.W); g = interior(c.gp, h)
  threaded_fill!(c.gp, 0, c.nthreads)
  resample_adjoint!(c.gp, c.kernel, c.binned, vals, c.Ñ, c.locks, c.B, c.nthreads)
  fold_halo!(c.gp, h, c.nthreads)
  apply!(c.bwd, g)
  resampling_correction_adjoint_core!(f, g, c.invphi, c.N, c.Ñ, c.nthreads)
  f
end

"""
    NFFTPlan{T,D,R}

NFFT plan of NFFT.jl, created by `plan_nfft(k, N; kwargs...)` or `NFFTPlan(k, N; kwargs...)`.

Generic options (every AbstractNFFTs implementation): `reltol` (default 1e-8), `dims`, `eltype`,
`storage`, `nthreads`. Options of NFFT.jl:

* `window`: `:auto` (default; all windows with a parameter table compete), `KaiserBessel()`,
  `ExpSemicircle()` or a `CustomWindow` (needs `W` and `σ`)
* `W`: window width (integer, odd allowed; W = 2m in the notation of Knopp et al. 2023)
* `σ`: oversampling factor
* `fft`: FFT backend, default `FFTWBackend()`
* `tilesize`: expert option, tile edge of the node binning
"""
mutable struct NFFTPlan{T,D,R} <: AbstractNFFTPlan{T,D,R}
  N::NTuple{D,Int}
  NOut::NTuple{R,Int}
  dims::UnitRange{Int}
  k::AbstractMatrix
  kernel::KernelEntry
  accuracy::Float64
  options::GenericOptions{T}
  fft::AbstractFFTBackend
  tilesize::Union{Nothing,Int}
  # how batch dimensions are handled: :single (none), :lead (batch dims before dims: batch-first
  # kernels), :trail (after dims: transposes into batch-first kernels), :loop (one transform per
  # slice; 3D, or batch dims on both sides)
  strategy::Symbol
  core::Any
  fbuf::Any                         # :trail: batch-first copies of input and output
  gbuf::Any
end

const NFFT_OPTIONS = (:window, :W, :σ, :fft, :tilesize)

function AbstractNFFTs.plan_nfft(b::NFFTBackend, k::AbstractMatrix, N::Dims{D}; kwargs...) where {D}
  g, rest = generic_options(k, N; kwargs...)
  check_options(b, rest, NFFT_OPTIONS)
  g.storage <: Array || throw(ArgumentError("storage = $(g.storage) is not supported yet"))
  NFFTPlan(k, N, g; rest...)
end

NFFTPlan(k::AbstractMatrix, N::Dims; kwargs...) = plan_nfft(NFFTBackend(), k, N; kwargs...)
NFFTPlan(k::AbstractVector, N; kwargs...) = plan_nfft(NFFTBackend(), k, N; kwargs...)

function NFFTPlan(k::AbstractMatrix, N::NTuple{D,Int}, g::GenericOptions{T}; window=:auto, W=nothing,
                  σ=nothing, fft=FFTWBackend(), tilesize=nothing) where {D,T}
  dims = g.dims
  Nd = N[dims]
  J = size(k, 2)
  if window isa CustomWindow
    (W === nothing || σ === nothing) && throw(ArgumentError("a CustomWindow needs W and σ"))
    kernel, acc = KernelEntry(window, W, σ, NaN, NaN, NaN, NaN), NaN
  elseif window === :auto || window isa AbstractWindow
    kernel, acc = choose_kernel(g.reltol, Nd, J, T; window, W, σ)
  else
    throw(ArgumentError("window must be :auto or an AbstractWindow, got $window"))
  end
  NOut = (N[1:first(dims)-1]..., J, N[last(dims)+1:end]...)
  strategy, core, fbuf, gbuf = make_engine(k, N, dims, kernel, T, fft, tilesize, g.nthreads)
  NFFTPlan{T,D,length(NOut)}(N, NOut, dims, k, kernel, acc, g, fft, tilesize, strategy, core, fbuf, gbuf)
end

batch_sizes(N, dims) = (prod(N[1:first(dims)-1]; init=1), prod(N[last(dims)+1:end]; init=1))

function make_engine(k, N, dims, kernel::KernelEntry, ::Type{T}, fft, tilesize, nthreads) where {T}
  Nd = N[dims]; Dd = length(dims); J = size(k, 2)
  L1, L3 = batch_sizes(N, dims)
  check_nodes(k)
  Ñ = grid_size(kernel.σ, Nd)
  φ = window_function(kernel.window, kernel.W, kernel.β, kernel.e0)
  if L1 == 1 && L3 == 1
    :single, NFFTCore(k, Nd, Ñ, kernel.W, φ, T, fft, tilesize, nthreads), nothing, nothing
  elseif Dd <= 2 && L1 > 1 && L3 == 1
    :lead, BatchCore(k, Nd, Ñ, kernel.W, φ, T, fft, L1, tilesize, nthreads), nothing, nothing
  elseif Dd <= 2 && L1 == 1
    :trail, BatchCore(k, Nd, Ñ, kernel.W, φ, T, fft, L3, tilesize, nthreads),
      Array{Complex{T}}(undef, L3, Nd...), Array{Complex{T}}(undef, L3, J)
  else
    :loop, NFFTCore(k, Nd, Ñ, kernel.W, φ, T, fft, tilesize, nthreads), nothing, nothing
  end
end

# Nodes must be finite and in [−1/2, 1/2] (1/2 is the same node as −1/2); anything else would be
# binned outside the grid.
function check_nodes(k)
  for x in k
    (isfinite(x) && -0.5 <= x <= 0.5) ||
      throw(ArgumentError("nodes must be finite and in [-1/2, 1/2], found $x"))
  end
end

# A copy shares the read-only parts (kernel, correction factors, binned nodes) and gets its own
# grid, FFT plans and locks, so that copies can run concurrently.
function Base.copy(c::NFFTCore{D}) where {D}
  gp = similar(c.gp)
  g = interior(gp, halo(c.W))
  fft = FFTWBackend()
  NFFTCore(c.N, c.Ñ, c.W, c.B, c.kernel, c.invphi, c.binned, gp,
           PassFFT(fft, g, c.N; dir=:fwd, nthreads=c.nthreads), PassFFT(fft, g, c.N; dir=:bwd, nthreads=c.nthreads),
           TileLocks(size(gp), c.B), c.nthreads)
end

function Base.copy(p::NFFTPlan{T,D,R}) where {T,D,R}
  strategy, core, fbuf, gbuf = p.strategy in (:single, :loop) ? (p.strategy, copy(p.core), nothing, nothing) :
    make_engine(p.k, p.N, p.dims, p.kernel, T, p.fft, p.tilesize, p.options.nthreads)
  NFFTPlan{T,D,R}(p.N, p.NOut, p.dims, p.k, p.kernel, p.accuracy, p.options, p.fft, p.tilesize,
                  strategy, core, fbuf, gbuf)
end

AbstractNFFTs.size_in(p::NFFTPlan) = p.N
AbstractNFFTs.size_out(p::NFFTPlan) = p.NOut
AbstractNFFTs.nodes(p::NFFTPlan) = p.k
AbstractNFFTs.accuracy(p::NFFTPlan) = p.accuracy

function AbstractNFFTs.nodes!(p::NFFTPlan{T}, k::AbstractMatrix) where {T}
  size(k, 1) == length(p.dims) || throw(DimensionMismatch("nodes must have $(length(p.dims)) rows"))
  J = size(k, 2)
  p.strategy, p.core, p.fbuf, p.gbuf = make_engine(k, p.N, p.dims, p.kernel, T, p.fft, p.tilesize, p.options.nthreads)
  p.k = k
  p.NOut = (p.N[1:first(p.dims)-1]..., J, p.N[last(p.dims)+1:end]...)
  p
end

"""
    NFFT.parameters(p::NFFTPlan) -> (; window, W, σ)

Window family, window width and oversampling factor the plan uses.
"""
parameters(p::NFFTPlan) = (window=p.kernel.window, W=p.kernel.W, σ=p.kernel.σ)

function Base.show(io::IO, p::NFFTPlan{T}) where {T}
  e = p.kernel
  print(io, "NFFTPlan{$T} for an input of size $(p.N) along dims $(p.dims) at $(size(p.k, 2)) nodes: ",
        "$(nameof(typeof(e.window))) window, W = $(e.W), σ = $(e.σ), predicted error $(p.accuracy)")
end

function check_sizes(p::NFFTPlan, f, fHat)
  size(f) == p.N || throw(DimensionMismatch("input has size $(size(f)), the plan expects $(p.N)"))
  size(fHat) == p.NOut || throw(DimensionMismatch("output has size $(size(fHat)), the plan expects $(p.NOut)"))
end

# Batch dimensions before and after `dims`: the arrays are viewed as (L1, N[dims]..., L3) and
# (L1, J, L3).
function batch_views(p::NFFTPlan, f, fHat)
  L1, L3 = batch_sizes(p.N, p.dims)
  reshape(f, L1, p.N[p.dims]..., L3), reshape(fHat, L1, size(p.k, 2), L3), L1, L3, length(p.dims)
end

function LinearAlgebra.mul!(fHat::AbstractArray, p::NFFTPlan{T,D}, f::AbstractArray) where {T,D}
  check_sizes(p, f, fHat)
  transform!(Val(p.strategy), p, p.core, fHat, f)
  fHat
end

function LinearAlgebra.mul!(f::AbstractArray, pl::Adjoint{Complex{T},<:NFFTPlan{T,D}}, fHat::AbstractArray) where {T,D}
  p = pl.parent
  check_sizes(p, f, fHat)
  transform_adjoint!(Val(p.strategy), p, p.core, f, fHat)
  f
end

# The core argument makes these the function barrier into code specialised on the kernel.
transform!(::Val{:single}, p, c, fHat, f) = trafo_core!(vec(fHat), c, f)
transform_adjoint!(::Val{:single}, p, c, f, fHat) = adjoint_core!(f, c, vec(fHat))

function transform!(::Val{:loop}, p, c, fHat, f)
  fr, gr, L1, L3, Dd = batch_views(p, f, fHat)
  for b in 1:L3, a in 1:L1
    trafo_core!(view(gr, a, :, b), c, view(fr, a, ntuple(_ -> Colon(), Dd)..., b))
  end
end
function transform_adjoint!(::Val{:loop}, p, c, f, fHat)
  fr, gr, L1, L3, Dd = batch_views(p, f, fHat)
  for b in 1:L3, a in 1:L1
    adjoint_core!(view(fr, a, ntuple(_ -> Colon(), Dd)..., b), c, view(gr, a, :, b))
  end
end

function transform!(::Val{:lead}, p, c, fHat, f)
  fr, gr, L1, L3, Dd = batch_views(p, f, fHat)
  trafo_batch!(view(gr, :, :, 1), c, view(fr, :, ntuple(_ -> Colon(), Dd)..., 1))
end
function transform_adjoint!(::Val{:lead}, p, c, f, fHat)
  fr, gr, L1, L3, Dd = batch_views(p, f, fHat)
  adjoint_batch!(view(fr, :, ntuple(_ -> Colon(), Dd)..., 1), c, view(gr, :, :, 1))
end

function transform!(::Val{:trail}, p, c, fHat, f)
  L3 = c.L
  transpose_ml!(reshape(p.fbuf, L3, :), reshape(f, :, L3), c.nthreads)
  trafo_batch!(p.gbuf, c, p.fbuf)
  transpose_lm!(reshape(fHat, :, L3), p.gbuf, c.nthreads)
end
function transform_adjoint!(::Val{:trail}, p, c, f, fHat)
  L3 = c.L
  transpose_ml!(p.gbuf, reshape(fHat, :, L3), c.nthreads)
  adjoint_batch!(p.fbuf, c, p.gbuf)
  transpose_lm!(reshape(f, :, L3), reshape(p.fbuf, L3, :), c.nthreads)
end

# Stage interface (Knopp et al. 2023), for plans without batch dimensions. `g` is an unpadded
# oversampled array of size oversampled_size(p). The FFT stages here are full (unpruned) FFTs, so
# they can be composed freely.
function full_core(p::NFFTPlan{T,D}) where {T,D}
  p.dims == 1:D || throw(ArgumentError("the stage interface needs a plan without batch dimensions"))
  p.core
end

check_size(a, sz, what) = size(a) == sz ||
  throw(DimensionMismatch("$what has size $(size(a)), the plan expects $sz"))

# the stage functions take (p, input, output) or (p, array)
function check_stage(p, f, g, fHat)
  c = full_core(p)
  f === nothing || check_size(f, p.N, "the equidistant array")
  g === nothing || check_size(g, c.Ñ, "the oversampled array")
  fHat === nothing || check_size(fHat, p.NOut, "the array at the nodes")
  c
end

AbstractNFFTs.oversampled_size(p::NFFTPlan) = full_core(p).Ñ

function AbstractNFFTs.resampling_correction!(p::NFFTPlan, f::AbstractArray, g::AbstractArray)
  c = check_stage(p, f, g, nothing)
  fill!(g, 0)
  resampling_correction_core!(g, f, c.invphi, c.N, c.Ñ, c.nthreads)
end

AbstractNFFTs.oversampled_fft!(p::NFFTPlan, g::AbstractArray) = (check_stage(p, nothing, g, nothing); FFTW.fft!(g); g)
AbstractNFFTs.oversampled_bfft!(p::NFFTPlan, g::AbstractArray) = (check_stage(p, nothing, g, nothing); FFTW.bfft!(g); g)

function AbstractNFFTs.resampling!(p::NFFTPlan, g::AbstractArray, fHat::AbstractArray)
  c = check_stage(p, nothing, g, fHat); h = halo(c.W)
  copyto!(interior(c.gp, h), g)
  fill_halo!(c.gp, h, c.nthreads)
  resample!(vec(fHat), c.gp, c.kernel, c.binned, c.Ñ, c.B, c.nthreads)
  fHat
end

function AbstractNFFTs.resampling_adjoint!(p::NFFTPlan, fHat::AbstractArray, g::AbstractArray)
  c = check_stage(p, nothing, g, fHat); h = halo(c.W)
  threaded_fill!(c.gp, 0, c.nthreads)
  resample_adjoint!(c.gp, c.kernel, c.binned, vec(fHat), c.Ñ, c.locks, c.B, c.nthreads)
  fold_halo!(c.gp, h, c.nthreads)
  copyto!(g, interior(c.gp, h))
end

function AbstractNFFTs.resampling_correction_adjoint!(p::NFFTPlan, g::AbstractArray, f::AbstractArray)
  c = check_stage(p, f, g, nothing)
  resampling_correction_adjoint_core!(f, g, c.invphi, c.N, c.Ñ, c.nthreads)
end
