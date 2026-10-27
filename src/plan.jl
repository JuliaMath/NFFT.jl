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
  # one extra point: tile assignment and support may round differently at tile edges
  bufs::Vector{Array{Complex{T},D}} # tile buffers of edge B+W+1
  flags::UInt32                     # FFTW flags, for copies
  nthreads::Int
end

# With fft = nothing a core for the resampling stages only (the spreading of the NNFFT): no FFT
# plans and no correction factors.
function NFFTCore(k::AbstractMatrix, N::NTuple{D,Int}, Ñ::NTuple{D,Int}, W, φ, ::Type{T}, fft, B, nthreads) where {D,T}
  K = make_kernel(φ, W, T)
  invphi = fft === nothing ? ntuple(_ -> T[], D) : inv_kernel_ft(φ, W, N, Ñ, T, nthreads)
  B = B === nothing ? default_tile(D, W) : B
  h = halo(W)
  gp = Array{Complex{T}}(undef, Ñ .+ W)         # zeroed by every transform
  g = interior(gp, h)
  NFFTCore(N, Ñ, W, B, K, invphi, Binned(k, Ñ, B, nthreads), gp,
           PassFFT(fft, g, N; dir=:fwd, nthreads), PassFFT(fft, g, N; dir=:bwd, nthreads),
           TileLocks(size(gp), B), task_buffers(Complex{T}, ntuple(_ -> B + W + 1, D), nthreads),
           something(fft, FFTWBackend()).flags, nthreads)
end

function trafo_core!(out, c::NFFTCore, f)
  h = halo(c.W); g = interior(c.gp, h)
  threaded_fill!(c.gp, 0, c.nthreads)
  resampling_correction_core!(g, f, c.invphi, c.N, c.Ñ, c.nthreads)
  apply!(c.fwd, g)
  fill_halo!(c.gp, h, c.nthreads)
  resample!(out, c.gp, c.kernel, c.binned, c.Ñ, c.bufs)
  out
end

function adjoint_core!(f, c::NFFTCore, vals)
  h = halo(c.W); g = interior(c.gp, h)
  threaded_fill!(c.gp, 0, c.nthreads)
  resample_adjoint!(c.gp, c.kernel, c.binned, vals, c.Ñ, c.locks, c.bufs)
  fold_halo!(c.gp, h, c.nthreads)
  apply!(c.bwd, g)
  resampling_correction_adjoint_core!(f, g, c.invphi, c.N, c.Ñ, c.nthreads)
  f
end

# Fields of the plans of NFFT, RNFFT, NFCT and NFST (shared methods in plans.jl).
macro plan_fields()
  esc(quote
    N::NTuple{D,Int}                  # array size (RNFFT: of the full coefficient array)
    Nin::NTuple{D,Int}                # input size (RNFFT: half along the first of dims)
    NOut::NTuple{R,Int}
    dims::UnitRange{Int}
    k::AbstractMatrix
    kernel::KernelEntry
    accuracy::Float64
    options::GenericOptions{T}
    fft::AbstractFFTBackend
    tilesize::Union{Nothing,Int}
    strategy::Symbol                  # batch dimensions, see make_engine
    core::Any
    fbuf::Any                         # :trail: batch-first copies of input and output
    gbuf::Any
  end)
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
  @plan_fields
end

function AbstractNFFTs.plan_nfft(b::NFFTBackend, k::AbstractMatrix, N::Dims{D}; kwargs...) where {D}
  g, rest = generic_options(k, N; kwargs...)
  check_options(b, rest, NFFT_OPTIONS)
  NFFTPlan(k, N, g; rest...)
end

NFFTPlan(k::AbstractMatrix, N::Dims; kwargs...) = plan_nfft(NFFTBackend(), k, N; kwargs...)
NFFTPlan(k::AbstractVector, N; kwargs...) = plan_nfft(NFFTBackend(), k, N; kwargs...)

NFFTPlan(k::AbstractMatrix, N::NTuple{D,Int}, g::GenericOptions; kwargs...) where {D} =
  fast_plan(NFFTPlan, k, N, g; kwargs...)

function make_core(::Type{<:NFFTPlan}, k, Nd, kernel::KernelEntry, ::Type{T}, fft, tilesize, nthreads, storage,
                   L1, L3) where {T}
  Ñ = grid_size(kernel.σ, Nd, kernel.W)
  φ = window_function(kernel)
  storage <: Array || return device_core(storage, k, Nd, Ñ, kernel.W, φ, T, nthreads, L1, L3)
  NFFTCore(k, Nd, Ñ, kernel.W, φ, T, fft, tilesize, nthreads)
end

slice_size(c::NFFTCore) = c.N

# 1D and 2D: batch dimensions only before dims run in the batch-first kernels, only after dims
# transposed into them.
batch_strategy(::Type{<:NFFTPlan}, Dd, L1, L3) = Dd > 2 ? :loop : L3 == 1 ? :lead : L1 == 1 ? :trail : :loop

function batch_engine(strategy, k, Nd, kernel::KernelEntry, ::Type{T}, fft, tilesize, nthreads, L1, L3) where {T}
  Ñ = grid_size(kernel.σ, Nd, kernel.W)
  c = BatchCore(k, Nd, Ñ, kernel.W, window_function(kernel), T, fft, L1 * L3, tilesize, nthreads)
  strategy === :lead && return :lead, c, nothing, nothing
  :trail, c, Array{Complex{T}}(undef, L3, Nd...), Array{Complex{T}}(undef, L3, size(k, 2))
end

# A copy shares the read-only parts (kernel, correction factors, binned nodes) and gets its own
# grid, FFT plans, locks and buffers, so that copies can run concurrently.
function copy_core(c::NFFTCore{D,K,T}, nthreads) where {D,K,T}
  gp = similar(c.gp)
  g = interior(gp, halo(c.W))
  fft = isempty(c.fwd.plans) ? nothing : FFTWBackend(flags=c.flags)
  NFFTCore(c.N, c.Ñ, c.W, c.B, c.kernel, c.invphi, c.binned, gp,
           PassFFT(fft, g, c.N; dir=:fwd, nthreads), PassFFT(fft, g, c.N; dir=:bwd, nthreads),
           TileLocks(size(gp), c.B), task_buffers(Complex{T}, size(c.bufs[1]), nthreads), c.flags, nthreads)
end

transform!(::Val{:lead}, p, c, fHat, f) = trafo_batch!(reshape(fHat, c.L, :), c, reshape(f, c.L, c.N...))
transform_adjoint!(::Val{:lead}, p, c, f, fHat) = adjoint_batch!(reshape(f, c.L, c.N...), c, reshape(fHat, c.L, :))

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

# the stage functions take (p, input, output) or (p, array)
function check_stage(p, f, g, fHat)
  c = full_core(p)
  f === nothing || check_size(f, p.N, "the equidistant array")
  g === nothing || check_size(g, c.Ñ, "the oversampled array")
  fHat === nothing || check_size(fHat, p.NOut, "the array at the nodes")
  c
end

AbstractNFFTs.oversampled_size(p::NFFTPlan) = full_core(p).Ñ

AbstractNFFTs.resampling_correction!(p::NFFTPlan, f::AbstractArray, g::AbstractArray) =
  (correction_stage!(check_stage(p, f, g, nothing), g, f); g)
AbstractNFFTs.oversampled_fft!(p::NFFTPlan, g::AbstractArray) = (check_stage(p, nothing, g, nothing); FFTW.fft!(g); g)
AbstractNFFTs.oversampled_bfft!(p::NFFTPlan, g::AbstractArray) = (check_stage(p, nothing, g, nothing); FFTW.bfft!(g); g)
AbstractNFFTs.resampling!(p::NFFTPlan, g::AbstractArray, fHat::AbstractArray) =
  (resampling_stage!(check_stage(p, nothing, g, fHat), vec(fHat), g); fHat)
AbstractNFFTs.resampling_adjoint!(p::NFFTPlan, fHat::AbstractArray, g::AbstractArray) =
  (resampling_adjoint_stage!(check_stage(p, nothing, g, fHat), g, vec(fHat)); g)
AbstractNFFTs.resampling_correction_adjoint!(p::NFFTPlan, g::AbstractArray, f::AbstractArray) =
  (correction_adjoint_stage!(check_stage(p, f, g, nothing), f, g); f)

function correction_stage!(c::NFFTCore, g, f)
  fill!(g, 0)
  resampling_correction_core!(g, f, c.invphi, c.N, c.Ñ, c.nthreads)
end

function resampling_stage!(c::NFFTCore, fHat, g)
  threaded_copy!(interior(c.gp, halo(c.W)), g, c.nthreads)
  resample_interior!(c, fHat)
end

resampling_adjoint_stage!(c::NFFTCore, g, fHat) = threaded_copy!(g, resample_adjoint_interior!(c, fHat), c.nthreads)

# The stages on the grid of the core itself: resampling from the values in the interior of the
# padded grid, and adjoint resampling into it (returns the interior as a view). The NNFFT uses them
# to pass the grid to its inner NFFT without a copy.
function resample_interior!(c::NFFTCore, fHat)
  fill_halo!(c.gp, halo(c.W), c.nthreads)
  resample!(fHat, c.gp, c.kernel, c.binned, c.Ñ, c.bufs)
end

function resample_adjoint_interior!(c::NFFTCore, fHat)
  h = halo(c.W)
  threaded_fill!(c.gp, 0, c.nthreads)
  resample_adjoint!(c.gp, c.kernel, c.binned, fHat, c.Ñ, c.locks, c.bufs)
  fold_halo!(c.gp, h, c.nthreads)
  interior(c.gp, h)
end

correction_adjoint_stage!(c::NFFTCore, f, g) =
  resampling_correction_adjoint_core!(f, g, c.invphi, c.N, c.Ñ, c.nthreads)
