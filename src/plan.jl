# The engine of one transform over the D NFFT dimensions (batch dimensions are handled by the
# plan). Its type depends on the kernel (width W is a compile-time constant of the hot loops); the
# plan holds it in an untyped field, so plan types stay inferable and every transform crosses one
# function barrier into the specialised loops.
struct NFFTCore{D,K<:SymKernel,T,Tk,PF<:PassFFT,PB<:PassFFT,FB<:Union{Nothing,AbstractFFTBackend}}
  N::NTuple{D,Int}
  Ñ::NTuple{D,Int}
  W::Int
  B::Int
  kernel::K
  invphi::NTuple{D,Vector{T}}
  binned::Binned{D,Tk}
  gp::Array{Complex{T},D}           # padded grid, padded_size(Ñ, W)
  zruns::Vector{Vector{UnitRange{Int}}}  # what the trafo zeroes, see unkept_runs
  fwd::PF                           # FFT passes of the backend
  bwd::PB
  locks::TileLocks{D}
  # one extra point: tile assignment and support may round differently at tile edges
  bufs::Vector{Array{Complex{T},D}} # tile buffers of edge B+W+1
  fft::FB                           # the backend, for copies
  nthreads::Int
end

# With fft = nothing a core for the resampling stages only (the spreading of the NNFFT): no FFT
# plans and no correction factors.
function NFFTCore(k::AbstractMatrix, N::NTuple{D,Int}, Ñ::NTuple{D,Int}, W, φ::F, ::Type{T}, fft, B, nthreads;
                  kernel=make_kernel(φ, W, T)) where {D,F,T}
  K = kernel
  invphi = fft === nothing ? ntuple(_ -> T[], D) : inv_kernel_ft(φ, W, N, Ñ, T, nthreads)
  B = B === nothing ? default_tile(D, W, size(k, 2) / prod(Ñ)) : B
  h = halo(W)
  gp = Array{Complex{T}}(undef, padded_size(Ñ, W))   # zeroed by every transform
  g = interior(gp, h, Ñ)
  binned = @stage :sort Binned(k, Ñ, B, nthreads)
  NFFTCore(N, Ñ, W, B, K, invphi, binned, gp, unkept_runs(N, Ñ, h, size(gp), nthreads),
           PassFFT(fft, g, N, Val(:fwd); nthreads), PassFFT(fft, g, N, Val(:bwd); nthreads),
           TileLocks(size(gp), B), task_buffers(Complex{T}, ntuple(_ -> B + W + 1, D), nthreads),
           fft, nthreads)
end

function trafo_core!(out, c::NFFTCore, f)
  h = halo(c.W); g = interior(c.gp, h, c.Ñ)
  @stage :zero zero_runs!(c.gp, c.zruns, c.nthreads)
  @stage :correction resampling_correction_core!(g, f, c.invphi, c.N, c.Ñ, c.nthreads)
  @stage :fft apply!(c.fwd, g)
  @stage :halo fill_halo!(c.gp, h, c.nthreads; Ñs=c.Ñ)
  @stage :resampling resample!(out, c.gp, c.kernel, c.binned, c.Ñ, c.bufs)
  out
end

function adjoint_core!(f, c::NFFTCore, vals)
  h = halo(c.W); g = interior(c.gp, h, c.Ñ)
  @stage :zero threaded_fill!(c.gp, 0, c.nthreads)
  @stage :resampling resample_adjoint!(c.gp, c.kernel, c.binned, vals, c.Ñ, c.locks, c.bufs)
  @stage :halo fold_halo!(c.gp, h, c.nthreads; Ñs=c.Ñ)
  @stage :fft apply!(c.bwd, g)
  @stage :correction resampling_correction_adjoint_core!(f, g, c.invphi, c.N, c.Ñ, c.nthreads)
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
    gridsize::Union{Nothing,Vector{Int}}  # Ñ per dimension of dims; nothing: from σ and W
    strategy::Symbol                  # batch dimensions, see make_engine
    core::Any                         # the engine of the strategy; behind it the transforms are type stable
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
* `fft`: FFT backend, `FFTWBackend()` (default) or `FFTABackend()` (`using FFTA`), see
  [`AbstractFFTBackend`](@ref)
* `tilesize`: expert option, tile edge of the node binning
* `gridsize`: expert option, size Ñ of the oversampled grid (integer, or one per transformed
  dimension), at least σN and W; any size, not only the FFT-friendly ones (2, 3, 5) the planner
  takes. The window stays the one designed for σ.
* `measure`: `true` times the four parameter sets of lowest predicted cost at plan
  time (transform and adjoint) and keeps the fastest; the cost model cannot see how fast the FFT
  is for a particular grid size

`T` is the real element type, `D` the number of dimensions of the array and `R` that of the output
(1 without batch dimensions). The compiler infers the type of the plan when `dims` is left out or
a single dimension; with a range it depends on its length.
"""
mutable struct NFFTPlan{T,D,R} <: AbstractNFFTPlan{T,D,R}
  @plan_fields
end

function AbstractNFFTs.plan_nfft(b::NFFTBackend, k::AbstractMatrix, N::Dims{D}; kwargs...) where {D}
  g, rest = generic_options(k, N; kwargs...)
  check_options(b, rest, NFFT_OPTIONS)
  fast_plan(NFFTPlan, k, N, g, out_ndims(N, kwargs); rest...)
end

NFFTPlan(k::AbstractMatrix, N::Dims; kwargs...) = plan_nfft(NFFTBackend(), k, N; kwargs...)
NFFTPlan(k::AbstractVector, N; kwargs...) = plan_nfft(NFFTBackend(), k, N; kwargs...)

NFFTPlan(k::AbstractMatrix, N::NTuple{D,Int}, g::GenericOptions; kwargs...) where {D} =
  fast_plan(NFFTPlan, k, N, g; kwargs...)

function make_core(::Type{<:NFFTPlan}, k, Nd, kernel::KernelEntry, ::Type{T}, fft, tilesize, gridsize, nthreads,
                   storage, L1, L3) where {T}
  Ñ = plan_grid(kernel, Nd, gridsize)
  φ = window_function(kernel)
  storage <: Array || return device_core(storage, k, Nd, Ñ, kernel.W, φ, T, nthreads, L1, L3)
  c = NFFTCore(k, Nd, Ñ, kernel.W, φ, T, fft, tilesize, nthreads)
  ABLATION[] === ABLATION_DEFAULT ? c : ablated_core(c, k)
end

slice_size(c::NFFTCore) = c.N

# 1D and 2D: batch dimensions only before dims run in the batch-first kernels, only after dims
# transposed into them.
batch_strategy(::Type{<:NFFTPlan}, Dd, L1, L3) = Dd > 2 ? :loop : L3 == 1 ? :lead : L1 == 1 ? :trail : :loop

function batch_engine(strategy, k, Nd, kernel::KernelEntry, ::Type{T}, fft, tilesize, gridsize, nthreads, L1,
                      L3) where {T}
  Ñ = plan_grid(kernel, Nd, gridsize)
  c = BatchCore(k, Nd, Ñ, kernel.W, window_function(kernel), T, fft, L1 * L3, tilesize, nthreads)
  strategy === :lead && return :lead, c
  :trail, TrailCore(c, Array{Complex{T}}(undef, L3, Nd...), Array{Complex{T}}(undef, L3, size(k, 2)))
end

# :trail: the batch core and batch-first copies of input and output
struct TrailCore{C<:BatchCore,A,B}
  core::C
  fbuf::A
  gbuf::B
end
Base.copy(t::TrailCore) = TrailCore(copy(t.core), similar(t.fbuf), similar(t.gbuf))

# A copy shares the read-only parts (kernel, correction factors, binned nodes) and gets its own
# grid, FFT plans, locks and buffers, so that copies can run concurrently.
function copy_core(c::NFFTCore{D,K,T}, nthreads) where {D,K,T}
  gp = similar(c.gp)
  g = interior(gp, halo(c.W), c.Ñ)
  NFFTCore(c.N, c.Ñ, c.W, c.B, c.kernel, c.invphi, c.binned, gp, unkept_runs(c.N, c.Ñ, halo(c.W), size(gp), nthreads),
           PassFFT(c.fft, g, c.N, Val(:fwd); nthreads), PassFFT(c.fft, g, c.N, Val(:bwd); nthreads),
           TileLocks(size(gp), c.B), task_buffers(Complex{T}, size(c.bufs[1]), nthreads), c.fft, nthreads)
end

transform!(::Val{:lead}, p, c, fHat, f) = trafo_batch!(reshape(fHat, c.L, :), c, reshape(f, c.L, c.N...))
transform_adjoint!(::Val{:lead}, p, c, f, fHat) = adjoint_batch!(reshape(f, c.L, c.N...), c, reshape(fHat, c.L, :))

function transform!(::Val{:trail}, p, t::TrailCore, fHat, f)
  c = t.core; L3 = c.L
  transpose_ml!(reshape(t.fbuf, L3, :), reshape(f, :, L3), c.nthreads)
  trafo_batch!(t.gbuf, c, t.fbuf)
  transpose_lm!(reshape(fHat, :, L3), t.gbuf, c.nthreads)
end
function transform_adjoint!(::Val{:trail}, p, t::TrailCore, f, fHat)
  c = t.core; L3 = c.L
  transpose_ml!(t.gbuf, reshape(fHat, :, L3), c.nthreads)
  adjoint_batch!(t.fbuf, c, t.gbuf)
  transpose_lm!(reshape(f, :, L3), reshape(t.fbuf, L3, :), c.nthreads)
end

# Stage interface (Knopp et al. 2023), for plans without batch dimensions. `g` is an unpadded
# oversampled array of size oversampled_size(p). The FFT stages here are full (unpruned) FFTs, so
# they can be composed freely.

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
AbstractNFFTs.oversampled_fft!(p::NFFTPlan, g::AbstractArray) = full_fft!(check_stage(p, nothing, g, nothing).fft, g, :fwd)
AbstractNFFTs.oversampled_bfft!(p::NFFTPlan, g::AbstractArray) = full_fft!(check_stage(p, nothing, g, nothing).fft, g, :bwd)
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

function resampling_stage!(c, fHat, g)
  threaded_copy!(interior(c.gp, halo(c.W), c.Ñ), g, c.nthreads)
  resample_interior!(c, fHat)
end

resampling_adjoint_stage!(c, g, fHat) = threaded_copy!(g, resample_adjoint_interior!(c, fHat), c.nthreads)

# The stages on the grid of a core (NFFTCore, RNFFTCore) itself: resampling from the values in the
# interior of the padded grid, and adjoint resampling into it (returns the interior as a view). The
# NNFFT uses them to pass the grid to its inner NFFT without a copy.
function resample_interior!(c, fHat)
  fill_halo!(c.gp, halo(c.W), c.nthreads; Ñs=c.Ñ)
  resample!(fHat, c.gp, c.kernel, c.binned, c.Ñ, c.bufs)
end

function resample_adjoint_interior!(c, fHat)
  h = halo(c.W)
  threaded_fill!(c.gp, 0, c.nthreads)
  resample_adjoint!(c.gp, c.kernel, c.binned, fHat, c.Ñ, c.locks, c.bufs)
  fold_halo!(c.gp, h, c.nthreads; Ñs=c.Ñ)
  interior(c.gp, h, c.Ñ)
end

correction_adjoint_stage!(c::NFFTCore, f, g) =
  resampling_correction_adjoint_core!(f, g, c.invphi, c.N, c.Ñ, c.nthreads)
