# What the plans of NFFT, RNFFT, NFCT and NFST share: construction from the generic and the
# NFFT.jl options, the strategy for batch dimensions, transforms, copies, new nodes and printing.
# A variant provides its core (`make_core`), the node range (`check_nodes`), the size the error
# model sees (`error_size`) and its input size (`input_size`).

const FastPlan{T,D,R} = Union{NFFTPlan{T,D,R},RNFFTPlan{T,D,R},NFCTPlan{T,D,R},NFSTPlan{T,D,R}}

const NFFT_OPTIONS = (:window, :W, :σ, :fft, :tilesize, :measure)

# Candidates timed with `measure = true`.
const MEASURE_CANDIDATES = 4

# Number of dimensions of the output, from the type of the `dims` keyword where that determines it
# (all dimensions, a single one), so that the type of the plan is inferred in these cases.
out_ndims(::Val{D}, ::Colon) where {D} = 1
out_ndims(::Val{D}, ::Integer) where {D} = D
out_ndims(::Val{D}, dims) where {D} = D - length(dims) + 1
out_ndims(::NTuple{D,Int}, kwargs) where {D} = Val(out_ndims(Val(D), get(kwargs, :dims, Colon())))

function fast_plan(::Type{P}, k::AbstractMatrix, N::NTuple{D,Int}, g::GenericOptions{T}, ::Val{R}=Val(D - length(g.dims) + 1);
                   window=:auto, W=nothing, σ=nothing, fft=nothing, tilesize=nothing, measure=false) where {P,D,T,R}
  fft = cpu_options(g.storage, fft, tilesize)
  check_backend(P, fft)
  dims = g.dims; Nd = N[dims]; J = size(k, 2)
  kernels = select_kernels(window, W, σ, g.reltol, error_size(P, Nd), J, T; Ncost=Nd,
                           n=measure ? MEASURE_CANDIDATES : 1, storage=g.storage)
  NOut = out_size(N, dims, J)
  build((kernel, acc)) = P{T,D,R}(N, input_size(P, N, dims), NOut, dims, to_storage(g.storage, k, T),
    kernel, acc, g, fft, tilesize, make_engine(P, k, N, dims, kernel, T, fft, tilesize, g.nthreads, g.storage)...)
  p = build(kernels[1])
  length(kernels) == 1 && return p
  # measure: the cost model cannot see how fast the FFT is for a particular grid size
  t = time_plan(p)
  for kernel in kernels[2:end]
    q = build(kernel)
    tq = time_plan(q)
    tq < t && ((p, t) = (q, tq))
  end
  p
end

# Element types of the input and the output of a transform.
data_eltypes(::NFFTPlan{T}) where {T} = (Complex{T}, Complex{T})
data_eltypes(::RNFFTPlan{T}) where {T} = (Complex{T}, T)
data_eltypes(::Union{NFCTPlan{T},NFSTPlan{T}}) where {T} = (T, T)

# Time of a transform and its adjoint (or transpose) on random data, the fastest of `nrep`.
function time_plan(p; nrep=3)
  Tin, Tout = data_eltypes(p)
  S = p.options.storage
  f = to_storage(S, randn(Tin, size_in(p))); fHat = to_storage(S, randn(Tout, size_out(p)))
  back = Tout <: Real && Tin <: Real ? transpose(p) : adjoint(p)
  run() = (mul!(fHat, p, f); mul!(f, back, fHat); synchronize(f))
  run()
  minimum(@elapsed(run()) for _ in 1:nrep)
end

# Wait for the transforms on a device to finish (GPU extension).
synchronize(x) = nothing

# Relative least squares (the small times weigh as much as the large ones) with nonnegative
# coefficients: every term is a cost; a negative coefficient would make extrapolation favour the
# large grids. Terms with negative coefficients are dropped one at a time, the most negative first.
function nonneg_fit(A, t)
  B = A ./ t
  keep = trues(size(A, 2))
  while true
    c = zeros(size(A, 2))
    c[keep] = B[:, keep] \ ones(length(t))
    all(>=(0), c) && return c
    keep[argmin(c)] = false
  end
end

const CALIBRATION_SIZES = (((2^16,), (2^20,)), ((256, 256), (1024, 1024)), ((32, 32, 32), (96, 96, 96)))

"""
    NFFT.calibrate_cost(storage=Array; eltype=Float64, J=(2^16, 2^18, 2^20), W=4:2:12,
                        σ=(1.25, 1.5, 2.0, 2.5), dims=1:3, sizes=CALIBRATION_SIZES, nrep=10, set=true)

Fit the cost model by which plans choose window width and oversampling to NFFTs timed on this
machine: transform and adjoint for random nodes and every combination of `J`, `W`, `σ` and the
sizes in `sizes[D]` (default 2^16 and 2^20, 256² and 1024², 32³ and 96³) in the dimensions
`dims`. With `set`, plans with this `storage` and `eltype` use the result for the rest of the
session. Returns a `Dict` from the dimension to the coefficients, for
[`NFFT.set_cost_model!`](@ref) in a later session.

The built-in models are fitted on an Apple M1 Max (CPU with 8 threads, GPU with Metal and
`Float32`). Other machines, thread counts and GPUs have other ratios of resampling, FFT and fixed
costs. The default sweep takes about three minutes on the M1 Max GPU; on the CPU, smaller `J`,
sizes and fewer `W` keep it short.
"""
function calibrate_cost(S=Array; eltype::Type{T}=Float64, J=(2^16, 2^18, 2^20), W=4:2:12,
                        σ=(1.25, 1.5, 2.0, 2.5), dims=1:3, sizes=CALIBRATION_SIZES, nrep=10, set=true) where {T}
  coef = Dict{Int,Any}()
  for D in dims
    rows = vec(map(Iterators.product(J, W, σ, sizes[D])) do (j, w, s, N)
      k = rand(D, j) .- 0.5
      p = plan_nfft(NFFTBackend(), k, N; eltype=T, storage=S, W=w, σ=s, reltol=1e-2)
      q = parameters(p)
      collect(cost_terms(S, D, q.W, prod(grid_size(q.σ, N)), j, T)), 1e3 * time_plan(p; nrep)
    end)
    A = reduce(vcat, permutedims(first(r)) for r in rows)
    t = [last(r) for r in rows]
    coef[D] = Tuple(nonneg_fit(A, t))
  end
  set && set_cost_model!(S, T, merge(Dict{Int,Any}(cost_model(S, T)), coef))
  coef
end

error_size(P, Nd) = Nd
input_size(P, N, dims) = N
out_size(N, dims, J) = (N[1:first(dims)-1]..., J, N[last(dims)+1:end]...)
# products of the sizes before and after dims (a loop, so that the result is inferred as Int)
function batch_sizes(N, dims)
  L1 = L3 = 1
  for d in eachindex(N)
    d < first(dims) && (L1 *= N[d])
    d > last(dims) && (L3 *= N[d])
  end
  L1, L3
end

# How batch dimensions are handled (`strategy`):
# - :single: no batch dimensions, or a device core that handles them itself
# - :slices: many or small slices in parallel on single-threaded cores (`core` is the vector of cores)
# - :loop: one transform per slice on the threaded core
# - :lead, :trail: the batch kernels of the NFFT (`batch_strategy`)
# The plan is set up on the CPU (parameters, binning, window); a device core moves what the
# transforms need to the device.
function make_engine(P, k, N, dims, kernel::KernelEntry, ::Type{T}, fft, tilesize, nthreads, storage) where {T}
  Nd = N[dims]; J = size(k, 2)
  L1, L3 = batch_sizes(N, dims); L = L1 * L3
  k = k isa Array ? k : Array(k)
  check_nodes(P, k)
  if !(storage <: Array) || L == 1
    return :single, make_core(P, k, Nd, kernel, T, fft, tilesize, nthreads, storage, L1, L3)
  end
  use_slices(L, J + prod(Nd), nthreads) &&
    return :slices, single_cores(make_core(P, k, Nd, kernel, T, fft, tilesize, nthreads, storage, 1, 1), L)
  strategy = batch_strategy(P, length(dims), L1, L3)
  strategy === :loop || return batch_engine(strategy, k, Nd, kernel, T, fft, tilesize, nthreads, L1, L3)
  :loop, make_core(P, k, Nd, kernel, T, fft, tilesize, nthreads, storage, 1, 1)
end

batch_strategy(P, Dd, L1, L3) = :loop

# Nodes must be finite and in [−1/2, 1/2] (1/2 is the same node as −1/2); anything else would be
# binned outside the grid.
function check_nodes(P, k)
  for x in k
    (isfinite(x) && -0.5 <= x <= 0.5) ||
      throw(ArgumentError("nodes must be finite and in [-1/2, 1/2], found $x"))
  end
end

# The FFT backend and the tile size are options of the CPU code; plans on a device reject them
# instead of ignoring them.
function cpu_options(storage, fft, tilesize)
  storage <: Array || (fft === nothing && tilesize === nothing) ||
    throw(ArgumentError("the options fft and tilesize are not supported with storage = $storage"))
  something(fft, FFTWBackend())
end

# The RNFFT, NFCT and NFST use FFTW's real and real-to-real transforms.
check_backend(P, fft) = nothing
check_backend(P::Type{<:Union{RNFFTPlan,NFCTPlan,NFSTPlan}}, fft) = fft isa FFTWBackend ||
  throw(ArgumentError("$(nameof(P)) needs the FFT backend FFTWBackend, got $(nameof(typeof(fft)))"))

# Core of a plan whose data live on a device (GPU array types); implemented by the GPU extension.
device_core(storage, args...) = throw(ArgumentError(
  "storage = $storage is not supported; for GPU arrays load the GPU package (CUDA, AMDGPU, Metal, …)"))

# The nodes as the plan returns them (`nodes(p)`): in the storage of the plan, on a device in the
# precision T of the plan (devices may lack Float64).
to_storage(::Type{S}, k) where {S} = k isa S ? k : S(Array(k))
to_storage(::Type{S}, k, ::Type{T}) where {S,T} = S <: Array || k isa S ? k : S(Array{T}(k))

AbstractNFFTs.size_in(p::FastPlan) = p.Nin
AbstractNFFTs.size_out(p::FastPlan) = p.NOut
AbstractNFFTs.nodes(p::FastPlan) = p.k
AbstractNFFTs.accuracy(p::FastPlan) = p.accuracy

"""
    NFFT.parameters(p) -> (; window, W, σ)

Window family, window width and oversampling factor the plan uses.
"""
parameters(p::FastPlan) = (window=p.kernel.window, W=p.kernel.W, σ=p.kernel.σ)

Base.copy(c::Union{NFFTCore,BatchCore,RNFFTCore,RealCore}) = copy_core(c, c.nthreads)

# A copy shares the read-only parts of its cores (kernel, correction factors, binned nodes) and
# gets its own work arrays, so that copies can run concurrently.
function Base.copy(p::FastPlan)
  core = p.strategy === :slices ? single_cores(p.core[1], length(p.core)) : copy(p.core)
  typeof(p)(p.N, p.Nin, p.NOut, p.dims, p.k, p.kernel, p.accuracy, p.options, p.fft, p.tilesize, p.strategy,
            core)
end

function AbstractNFFTs.nodes!(p::FastPlan{T}, k::AbstractMatrix) where {T}
  size(k, 1) == length(p.dims) || throw(DimensionMismatch("nodes must have $(length(p.dims)) rows"))
  p.strategy, p.core = make_engine(typeof(p), k, p.N, p.dims, p.kernel, T, p.fft, p.tilesize,
                                                   p.options.nthreads, p.options.storage)
  p.k = to_storage(p.options.storage, k, T)
  p.NOut = out_size(p.N, p.dims, size(k, 2))
  p
end

function Base.show(io::IO, p::FastPlan{T}) where {T}
  e = p.kernel
  print(io, "$(nameof(typeof(p))){$T} of size $(p.N) along dims $(p.dims) at $(size(p.k, 2)) nodes: ",
        "$(nameof(typeof(e.window))) window, W = $(e.W), σ = $(e.σ), predicted error $(p.accuracy)")
end

# the core of a plan without batch dimensions, for the stage interface
function full_core(p::FastPlan{T,D}) where {T,D}
  p.dims == 1:D || throw(ArgumentError("the stage interface needs a plan without batch dimensions"))
  p.core
end

check_size(a, sz, what) = size(a) == sz ||
  throw(DimensionMismatch("$what has size $(size(a)), the plan expects $sz"))
check_sizes(p, f, fHat) = (check_size(f, p.Nin, "input"); check_size(fHat, p.NOut, "output"))

function LinearAlgebra.mul!(fHat::AbstractArray, p::FastPlan, f::AbstractArray)
  check_sizes(p, f, fHat)
  transform!(Val(p.strategy), p, p.core, fHat, f)
  fHat
end

# the adjoint of the complex transforms, the transpose of the real ones
for A in (:(Adjoint{Complex{T},<:NFFTPlan{T}}), :(Adjoint{Complex{T},<:RNFFTPlan{T}}),
          :(Transpose{T,<:NFCTPlan{T}}), :(Transpose{T,<:NFSTPlan{T}}))
  @eval function LinearAlgebra.mul!(f::AbstractArray, pl::$A, fHat::AbstractArray) where {T}
    p = parent(pl)
    check_sizes(p, f, fHat)
    transform_adjoint!(Val(p.strategy), p, p.core, f, fHat)
    f
  end
end

# The core argument makes these the function barrier into code specialised on the kernel.
transform!(::Val{:single}, p, c, fHat, f) = trafo_core!(vec(fHat), c, f)
transform_adjoint!(::Val{:single}, p, c, f, fHat) = adjoint_core!(f, c, vec(fHat))

# Batch dimensions before and after `dims`: the arrays are viewed as (L1, slice..., L3) and
# (L1, J, L3); `slice_size(c)` is the input size of one transform of the core.
function slice_views(p, c, f, fHat)
  L1, L3 = batch_sizes(p.N, p.dims)
  S = slice_size(c)
  reshape(f, L1, S..., L3), reshape(fHat, L1, :, L3), L1, L3, ntuple(_ -> Colon(), Val(length(S)))
end

function transform!(::Val{:loop}, p, c, fHat, f)
  fr, gr, L1, L3, cs = slice_views(p, c, f, fHat)
  for b in 1:L3, a in 1:L1
    trafo_core!(view(gr, a, :, b), c, view(fr, a, cs..., b))
  end
end
function transform_adjoint!(::Val{:loop}, p, c, f, fHat)
  fr, gr, L1, L3, cs = slice_views(p, c, f, fHat)
  for b in 1:L3, a in 1:L1
    adjoint_core!(view(fr, a, cs..., b), c, view(gr, a, :, b))
  end
end

function transform!(::Val{:slices}, p, cores, fHat, f)
  fr, gr, L1, L3, cs = slice_views(p, cores[1], f, fHat)
  slices_parallel(cores, L1, L3) do c, a, b
    trafo_core!(view(gr, a, :, b), c, view(fr, a, cs..., b))
  end
end
function transform_adjoint!(::Val{:slices}, p, cores, f, fHat)
  fr, gr, L1, L3, cs = slice_views(p, cores[1], f, fHat)
  slices_parallel(cores, L1, L3) do c, a, b
    adjoint_core!(view(fr, a, cs..., b), c, view(gr, a, :, b))
  end
end
