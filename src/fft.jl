"""
    AbstractFFTBackend

FFT used for the oversampled grid by NFFT plans (and the NNFFT): batched 1D transforms along one
dimension of an array view, in place as seen from NFFT.jl. A backend `B` implements

* `NFFT.pass_plan(::B, ::Val{K}, v, d)`: a plan for the transform of the view `v` along dimension
  `d`, forward for `K = -1`, unnormalised backward for `K = 1`
* `NFFT.apply_pass!(plan, v)`: apply it to `v` in place (default: `plan * v`, an in-place plan)
* optionally `NFFT.planning(f, ::B, nthreads)`: run the planning `f()` with `nthreads` threads per
  plan (default: `f()`)

The RNFFT, NFCT and NFST use FFTW's real and real-to-real transforms and need [`FFTWBackend`](@ref).
"""
abstract type AbstractFFTBackend end

"""
    FFTWBackend(; flags = FFTW.ESTIMATE)

FFTW on the CPU, the default. In D ≥ 2 NFFT.jl threads the per-dimension passes itself over
single-threaded FFTW plans: FFTW's own threading with ESTIMATE produces plans up to 17× slower than
MEASURE, while single-threaded batched ESTIMATE plans are within 10–20 % of MEASURE. A transform with
one dimension uses FFTW's own threading.
"""
Base.@kwdef struct FFTWBackend <: AbstractFFTBackend
  flags::UInt32 = FFTW.ESTIMATE
end

"""
    FFTABackend()

[FFTA.jl](https://github.com/JuliaMath/FFTA.jl), an FFT in pure Julia; needs `using FFTA`. For NFFT
and NNFFT plans. Slower than FFTW; its plans work out of place, so every pass keeps a buffer.
"""
struct FFTABackend <: AbstractFFTBackend end

pass_plan(fft::AbstractFFTBackend, args...) =
  throw(ArgumentError("the FFT backend $(nameof(typeof(fft))) is not available; for FFTABackend load FFTA"))
apply_pass!(plan, v) = (plan * v; nothing)
planning(f, ::AbstractFFTBackend, nthreads) = f()

pass_plan(fft::FFTWBackend, ::Val{K}, v, d) where {K} =
  K == FFTW.FORWARD ? FFTW.plan_fft!(v, d; flags=fft.flags) : FFTW.plan_bfft!(v, d; flags=fft.flags)
planning(f, ::FFTWBackend, nthreads) = with_fftw_threads(f, nthreads)

# Threads for an FFTW plan threaded by FFTW itself (a single 1D pass) over n points. They pay off
# from about 2^14 points (M1 Max, 10 threads: 8192 points twice as slow, 16384 1.2× and 32768 1.9×
# faster) and make planning 5–10× slower (0.15–0.5 ms per plan).
fft_nthreads(n, nthreads) = n >= 2^14 ? nthreads : 1

# The thread count of FFTW's planner is global; planning restores it.
function with_fftw_threads(f, n)
  old = FFTW.get_num_threads()
  FFTW.set_num_threads(n)
  try
    f()
  finally
    FFTW.set_num_threads(old)
  end
end

# One pass per transformed dimension, each over a set of boxes (array views) with a batched 1D
# plan. Forward (direct NFFT): the input is nonzero only in the corners of the kept modes, passes
# run along the transformed dims in decreasing order, pass d only on lines whose transformed dims
# < d are kept. Backward (adjoint NFFT): only the corner outputs are needed, passes in increasing
# order with the same restriction. Batch dimensions (not in `tdims`) are never transformed.
struct PassFFT{D,P}
  plans::Vector{Vector{P}}
  boxes::Vector{Vector{NTuple{D,UnitRange{Int}}}}
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
  PassFFT(plans, boxes)
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
  PassFFT{D,Nothing}(Vector{Nothing}[], Vector{NTuple{D,UnitRange{Int}}}[])

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
    @maybe_threads length(ps) :dynamic for i in eachindex(ps)
      apply_pass!(ps[i], view(x, bs[i]...))
    end
  end
  x
end
