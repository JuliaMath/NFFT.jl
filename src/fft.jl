"""
    AbstractFFTBackend

FFT used for the oversampled grid by NFFT plans (and the NNFFT): batched 1D transforms along one
dimension of an array view, in place as seen from NFFT.jl. A backend `B` implements

* `NFFT.pass_plan(::B, ::Val{K}, v, d)`: a plan for the transform of the view `v` along dimension
  `d`, forward for `K = -1`, unnormalised backward for `K = 1`
* `NFFT.apply_pass!(plan, v)`: apply it to `v` in place (default: `plan * v`, an in-place plan)
* optionally `NFFT.planning(f, ::B, nthreads)`: run the planning `f()` with `nthreads` threads per
  plan (default: `f()`)

The real transforms (RNFFT, NFCT, NFST) run with every backend: FFTW has native real and
real-to-real plans, the others through complex FFTs of the same length (`ComplexRealFFT`) and
FFTL's real-to-real passes.
"""
abstract type AbstractFFTBackend end

"""
    FFTLBackend()

[FFTL.jl](https://github.com/JuliaMath/FFTL.jl), the default backend: an FFT in pure Julia with
batched in-place transforms along one dimension of strided views, on the CPU through `plan_pass`
and on GPU arrays through the KernelAbstractions extension. The first plan of a length compiles its
kernels.
"""
struct FFTLBackend <: AbstractFFTBackend end

"""
    FFTWBackend(; flags = 1 << 6)

FFTW on the CPU. In D ≥ 2 NFFT.jl threads the per-dimension passes itself over single-threaded FFTW
plans: FFTW's own threading with ESTIMATE produces plans up to 17× slower than MEASURE, while
single-threaded batched ESTIMATE plans are within 10–20 % of MEASURE. A transform with one dimension
uses FFTW's own threading. `flags` is FFTW's planning flag, `1 << 6` is `FFTW.ESTIMATE`; needs
`using FFTW` (the backend is a weak dependency).
"""
Base.@kwdef struct FFTWBackend <: AbstractFFTBackend
  flags::UInt32 = 1 << 6    # FFTW.ESTIMATE (the ABI value; a weak dependency, so the numeric value)
end

"""
    FFTABackend()

[FFTA.jl](https://github.com/JuliaMath/FFTA.jl), an FFT in pure Julia; needs `using FFTA`. For NFFT
and NNFFT plans. Slower than FFTL and FFTW; its plans work out of place, so every pass keeps a
buffer.
"""
struct FFTABackend <: AbstractFFTBackend end

pass_plan(fft::AbstractFFTBackend, args...) =
  throw(ArgumentError("the FFT backend $(nameof(typeof(fft))) is not available; load FFTW for FFTWBackend, FFTA for FFTABackend"))
apply_pass!(plan, v) = (plan * v; nothing)
planning(f, ::AbstractFFTBackend, nthreads) = f()

# FFTL (the default, a hard dependency): the pass interface. Task-local thread count, so that plans
# made concurrently on other tasks keep their own.
planning(f, ::FFTLBackend, nthreads) = task_local_storage(f, :nfft_fftl_threads, nthreads)
pass_plan(::FFTLBackend, ::Val{K}, v, d) where {K} =
  FFTL.plan_pass(v, d; sign=K, threads=get(task_local_storage(), :nfft_fftl_threads, 1))
apply_pass!(p::FFTL.PassPlan, v) = (FFTL.apply!(p, v); nothing)

# the real transforms of the RNFFT as native r2c/c2r passes instead of the complex fallback of the
# same length. The released FFTL (0.1.0) has no r2c/c2r passes yet — against it the complex fallback
# stays; FFTL main has them since the realpass merge (PR #8), with the strides of input and output
# read in mul! from the applied arrays (so the strided interior views of the RNFFT's real grid run).
if isdefined(FFTL, :plan_r2c_pass)
  rfft_plan(::FFTLBackend, x, X) =
    FFTL.plan_r2c_pass(x, 1; threads=get(task_local_storage(), :nfft_fftl_threads, 1))
  brfft_plan(::FFTLBackend, X, x) =
    FFTL.plan_c2r_pass(X, 1; n=size(x, 1), threads=get(task_local_storage(), :nfft_fftl_threads, 1))
end

# The full-transform shortcuts of the RNFFT stage interface: one call instead of the pass machinery.
# The generic backend has none; the backends with an AbstractFFTs interface do.
full_brfft(fft::AbstractFFTBackend, X, n) = nothing
full_brfft(fft::FFTLBackend, X, n) = FFTL.brfft(X, n)
full_rfft(fft::AbstractFFTBackend, g, n) = nothing
full_rfft(fft::FFTLBackend, g, n) = FFTL.rfft(g)

# Threads for a 1D pass threaded by the backend itself over n points. They pay off from about 2^14
# points (M1 Max, 10 threads: 8192 points twice as slow, 16384 1.2× and 32768 1.9× faster) and make
# planning 5–10× slower (0.15–0.5 ms per plan).
fft_nthreads(n, nthreads) = n >= 2^14 ? nthreads : 1
# One pass per transformed dimension, each over a set of boxes (array views) with a batched 1D
# plan. Forward (direct NFFT): the input is nonzero only in the corners of the kept modes, passes
# run along the transformed dims in decreasing order, pass d only on lines whose transformed dims
# < d are kept. Backward (adjoint NFFT): only the corner outputs are needed, passes in increasing
# order with the same restriction. Batch dimensions (not in `tdims`) are never transformed.
struct PassFFT{D,P}
  plans::Vector{Vector{P}}
  boxes::Vector{Vector{NTuple{D,UnitRange{Int}}}}
  nthreads::Int                     # tasks per pass at most
end
# Plans whose type depends on the dimension (FFTL): their common supertype, one dynamic call per box.
# `plans` is an untyped `Vector{Any}` when no pass remains (the real transforms in 1D) or the plan
# type does not infer across dimensions.
function PassFFT(plans::Vector, boxes::Vector{Vector{NTuple{D,UnitRange{Int}}}}, nthreads) where {D}
  P = mapreduce(eltype, typejoin, plans; init=Union{})
  PassFFT{D,P}([convert(Vector{P}, p) for p in plans], boxes, nthreads)
end

# `Val(dir)` selects the transform (:fwd forward, :bwd unnormalised backward; a type, so that the
# plan type is inferred), `pattern` the pruning (:fwd zero input outside the kept modes, :bwd only
# the kept outputs needed); they differ for the RNFFT. With `fft_threads`, a single pass is
# threaded by the backend (FFTW) instead of split into boxes.
function PassFFT(fft::AbstractFFTBackend, x::AbstractArray{C,D}, N::NTuple{D,Int}, ::Val{dir}; pattern=dir,
                 nthreads=Threads.nthreads(), tdims=1:D, fft_threads=length(tdims) == 1) where {C,D,dir}
  K = dir === :fwd ? Val(-1) : Val(1)
  order = pattern == :fwd ? reverse(tdims) : tdims
  boxes = [pass_boxes(x, N, d, tdims, fft_threads, nthreads) for d in order]
  plans = planning(fft, fft_threads ? fft_nthreads(length(x), nthreads) : 1) do
    [pass_plans(fft, K, [view(x, b...) for b in bs], d) for (d, bs) in zip(order, boxes)]
  end
  PassFFT(plans, boxes, nthreads)
end

# The plans of one pass, one per box. FFTW executes a plan on any array of the same size, strides
# and alignment, also from several threads at once, so boxes of the same layout share a plan
# (planning 20–32 boxes is most of the FFT setup in 2D and 3D). Other backends may keep state in
# the plan (FFTA: a buffer).
pass_plans(fft::AbstractFFTBackend, K, vs, d) = [pass_plan(fft, K, v, d) for v in vs]
function pass_plans(fft::FFTWBackend, K, vs, d)
  isempty(vs) && return [pass_plan(fft, K, v, d) for v in vs]
  layout(v) = (size(v), strides(v), UInt(pointer(v)) % 16)
  plans = [pass_plan(fft, K, vs[1], d)]
  for i in 2:length(vs)
    j = findfirst(j -> layout(vs[j]) == layout(vs[i]), 1:i-1)
    push!(plans, j === nothing ? pass_plan(fft, K, vs[i], d) : plans[j])
  end
  plans
end

# The boxes of the pass along d: the kept index ranges along the transformed dims < d, everything
# else; split along the outermost dimension other than d for threading.
function pass_boxes(x::AbstractArray{C,D}, N, d, tdims, fft_threads, nthreads) where {C,D}
  Ñ = size(x)
  sets = ntuple(k -> (k < d && k in tdims) ? UnitRange{Int}[r for r in keptranges(N[k], Ñ[k]) if !isempty(r)] :
                                             UnitRange{Int}[1:Ñ[k]], Val(D))
  bxs = vec(collect(Iterators.product(sets...)))
  s = fft_threads ? 0 : d == D ? D - 1 : D
  s == 0 && return bxs
  pb = NTuple{D,UnitRange{Int}}[]
  for b in bxs, c in chunks(b[s], cld(nthreads, length(bxs)))
    push!(pb, Base.setindex(b, c, s))
  end
  pb
end

# No FFT: the core of the NNFFT spreading only resamples.
PassFFT(::Nothing, x::AbstractArray{C,D}, N, dir; kwargs...) where {C,D} =
  PassFFT{D,Nothing}(Vector{Nothing}[], Vector{NTuple{D,UnitRange{Int}}}[], 1)

# The full FFT of `g` along all dimensions (the FFT stages of the stage interface).
function full_fft!(fft, g::AbstractArray{C,D}, dir) where {C,D}
  K = dir == :fwd ? Val(-1) : Val(1)
  for d in 1:D
    apply_pass!(planning(() -> pass_plan(fft, K, g, d), fft, 1), g)
  end
  g
end

function apply!(P::PassFFT, x)
  for (ps, bs) in zip(P.plans, P.boxes)
    @maybe_threads min(P.nthreads, length(ps)) :dynamic for i in eachindex(ps)
      apply_pass!(ps[i], view(x, bs[i]...))
    end
  end
  x
end

# Real FFTs along dimension 1 of a real array x of length n there: r2c to the half spectrum
# (n÷2 + 1 points) and its inverse c2r (unnormalised). FFTW has them as plans (the extension);
# other backends get them from a complex FFT of the same length (twice the work), the c2r with the
# spectrum completed by Hermitian symmetry. Applied with mul!(out, plan, in).

struct ComplexRealFFT{K,P,C}
  plan::P
  buf::C          # complex copy of the full lines
end
function complex_real_fft(fft, ::Val{K}, x) where {K}
  buf = Array{complex(eltype(x))}(undef, size(x))
  ComplexRealFFT{K,typeof(pass_plan(fft, Val(K), buf, 1)),typeof(buf)}(pass_plan(fft, Val(K), buf, 1), buf)
end
rfft_plan(fft::AbstractFFTBackend, x, X) = complex_real_fft(fft, Val(-1), x)
brfft_plan(fft::AbstractFFTBackend, X, x) = complex_real_fft(fft, Val(1), x)

function LinearAlgebra.mul!(X::AbstractArray, p::ComplexRealFFT{-1}, x::AbstractArray)
  copyto!(p.buf, x)
  apply_pass!(p.plan, p.buf)
  copyto!(X, view(p.buf, axes(X, 1), ntuple(_ -> Colon(), ndims(X) - 1)...))
end
function LinearAlgebra.mul!(x::AbstractArray, p::ComplexRealFFT{1}, X::AbstractArray)
  n = size(p.buf, 1); h = size(X, 1)
  rest = ntuple(_ -> Colon(), ndims(X) - 1)
  copyto!(view(p.buf, 1:h, rest...), X)
  view(p.buf, h+1:n, rest...) .= conj.(view(X, n-h+1:-1:2, rest...))
  apply_pass!(p.plan, p.buf)
  x .= real.(p.buf)
end
