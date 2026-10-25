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
  core::Any
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
  core = make_core(k, Nd, kernel, T, fft, tilesize, g.nthreads)
  NOut = (N[1:first(dims)-1]..., J, N[last(dims)+1:end]...)
  NFFTPlan{T,D,length(NOut)}(N, NOut, dims, k, kernel, acc, g, fft, tilesize, core)
end

function make_core(k, Nd, kernel::KernelEntry, ::Type{T}, fft, tilesize, nthreads) where {T}
  Ñ = grid_size(kernel.σ, Nd)
  φ = window_function(kernel.window, kernel.W, kernel.β, kernel.e0)
  NFFTCore(k, Nd, Ñ, kernel.W, φ, T, fft, tilesize, nthreads)
end

AbstractNFFTs.size_in(p::NFFTPlan) = p.N
AbstractNFFTs.size_out(p::NFFTPlan) = p.NOut
AbstractNFFTs.nodes(p::NFFTPlan) = p.k
AbstractNFFTs.accuracy(p::NFFTPlan) = p.accuracy

function AbstractNFFTs.nodes!(p::NFFTPlan{T}, k::AbstractMatrix) where {T}
  size(k, 1) == length(p.dims) || throw(DimensionMismatch("nodes must have $(length(p.dims)) rows"))
  J = size(k, 2)
  p.k = k
  p.core = make_core(k, p.N[p.dims], p.kernel, T, p.fft, p.tilesize, p.options.nthreads)
  p.NOut = (p.N[1:first(p.dims)-1]..., J, p.N[last(p.dims)+1:end]...)
  p
end

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
# (L1, J, L3), and every (a, c) slice is one transform.
function batch_views(p::NFFTPlan{T,D}, f, fHat) where {T,D}
  Nd = p.N[p.dims]; J = size(p.k, 2)
  L1 = prod(p.N[1:first(p.dims)-1]; init=1); L3 = prod(p.N[last(p.dims)+1:end]; init=1)
  reshape(f, L1, Nd..., L3), reshape(fHat, L1, J, L3), L1, L3, length(Nd)
end

function LinearAlgebra.mul!(fHat::AbstractArray, p::NFFTPlan{T,D}, f::AbstractArray) where {T,D}
  check_sizes(p, f, fHat)
  if p.dims == 1:D
    trafo_core!(vec(fHat), p.core, f)
  else
    fr, gr, L1, L3, Dd = batch_views(p, f, fHat)
    for c in 1:L3, a in 1:L1
      trafo_core!(view(gr, a, :, c), p.core, view(fr, a, ntuple(_ -> Colon(), Dd)..., c))
    end
  end
  fHat
end

function LinearAlgebra.mul!(f::AbstractArray, pl::Adjoint{Complex{T},<:NFFTPlan{T,D}}, fHat::AbstractArray) where {T,D}
  p = pl.parent
  check_sizes(p, f, fHat)
  if p.dims == 1:D
    adjoint_core!(f, p.core, vec(fHat))
  else
    fr, gr, L1, L3, Dd = batch_views(p, f, fHat)
    for c in 1:L3, a in 1:L1
      adjoint_core!(view(fr, a, ntuple(_ -> Colon(), Dd)..., c), p.core, view(gr, a, :, c))
    end
  end
  f
end

# Stage interface (Knopp et al. 2023), for plans without batch dimensions. `g` is an unpadded
# oversampled array of size oversampled_size(p). The FFT stages here are full (unpruned) FFTs, so
# they can be composed freely.
function full_core(p::NFFTPlan{T,D}) where {T,D}
  p.dims == 1:D || throw(ArgumentError("the stage interface needs a plan without batch dimensions"))
  p.core
end

AbstractNFFTs.oversampled_size(p::NFFTPlan) = full_core(p).Ñ

function AbstractNFFTs.resampling_correction!(p::NFFTPlan, f::AbstractArray, g::AbstractArray)
  c = full_core(p)
  fill!(g, 0)
  resampling_correction_core!(g, f, c.invphi, c.N, c.Ñ, c.nthreads)
end

AbstractNFFTs.oversampled_fft!(p::NFFTPlan, g::AbstractArray) = (full_core(p); FFTW.fft!(g); g)
AbstractNFFTs.oversampled_bfft!(p::NFFTPlan, g::AbstractArray) = (full_core(p); FFTW.bfft!(g); g)

function AbstractNFFTs.resampling!(p::NFFTPlan, g::AbstractArray, fHat::AbstractArray)
  c = full_core(p); h = halo(c.W)
  copyto!(interior(c.gp, h), g)
  fill_halo!(c.gp, h, c.nthreads)
  resample!(vec(fHat), c.gp, c.kernel, c.binned, c.Ñ, c.B, c.nthreads)
  fHat
end

function AbstractNFFTs.resampling_adjoint!(p::NFFTPlan, fHat::AbstractArray, g::AbstractArray)
  c = full_core(p); h = halo(c.W)
  threaded_fill!(c.gp, 0, c.nthreads)
  resample_adjoint!(c.gp, c.kernel, c.binned, vec(fHat), c.Ñ, c.locks, c.B, c.nthreads)
  fold_halo!(c.gp, h, c.nthreads)
  copyto!(g, interior(c.gp, h))
end

function AbstractNFFTs.resampling_correction_adjoint!(p::NFFTPlan, g::AbstractArray, f::AbstractArray)
  c = full_core(p)
  resampling_correction_adjoint_core!(f, g, c.invphi, c.N, c.Ñ, c.nthreads)
end
