"""
    AbstractFFTBackend

FFT used for the oversampled grid. A backend must provide in-place batched 1D transforms along a
chosen dimension of an array view (forward and unnormalised backward).
"""
abstract type AbstractFFTBackend end

"""
    FFTWBackend(; flags = FFTW.ESTIMATE)

FFTW on the CPU. In D ≥ 2 NFFT.jl threads the per-dimension passes itself over single-threaded
FFTW plans: FFTW's own threading with ESTIMATE produces plans up to 17× slower than MEASURE, while
single-threaded batched ESTIMATE plans are within 10–20 % of MEASURE. A transform with one
dimension uses FFTW's own threading.
"""
Base.@kwdef struct FFTWBackend <: AbstractFFTBackend
  flags::UInt32 = FFTW.ESTIMATE
end

# One pass per transformed dimension, each over a set of boxes (array views) with a batched 1D
# plan. Forward (direct NFFT): the input is nonzero only in the corners of the kept modes, passes
# run along the transformed dims in decreasing order, pass d only on lines whose transformed dims
# < d are kept. Backward (adjoint NFFT): only the corner outputs are needed, passes in increasing
# order with the same restriction. Batch dimensions (not in `tdims`) are never transformed.
struct PassFFT{D}
  plans::Vector{Vector{Any}}
  boxes::Vector{Vector{NTuple{D,UnitRange{Int}}}}
end

# `dir` selects the transform (:fwd forward, :bwd unnormalised backward), `pattern` the pruning
# (:fwd zero input outside the kept modes, :bwd only the kept outputs needed); they differ for the
# RNFFT. With `fftw_threads`, a single pass is threaded by FFTW instead of split into boxes.
function PassFFT(fft::FFTWBackend, x::AbstractArray{T,D}, N::NTuple{D,Int}; dir=:fwd, pattern=dir,
                 prune=true, nthreads=Threads.nthreads(), tdims=1:D,
                 fftw_threads=length(tdims) == 1) where {T,D}
  Ñ = size(x)
  onepass = fftw_threads
  FFTW.set_num_threads(onepass ? nthreads : 1)
  order = pattern == :fwd ? reverse(tdims) : tdims
  plans, boxes = Vector{Any}[], Vector{NTuple{D,UnitRange{Int}}}[]
  for d in order
    sets = ntuple(k -> (prune && k < d && k in tdims) ? filter(!isempty, keptranges(N[k], Ñ[k])) : (1:Ñ[k],), D)
    bxs = vec(collect(Iterators.product(sets...)))
    # split each box along the outermost dimension other than d
    s = onepass ? 0 : d == D ? D - 1 : D
    pb = s == 0 ? bxs : NTuple{D,UnitRange{Int}}[]
    s == 0 || for b in bxs, c in chunks(b[s], cld(nthreads, length(bxs)))
      push!(pb, Base.setindex(b, c, s))
    end
    push!(boxes, pb)
    push!(plans, Any[dir == :fwd ? FFTW.plan_fft!(view(x, b...), d; flags=fft.flags) :
                                   FFTW.plan_bfft!(view(x, b...), d; flags=fft.flags) for b in pb])
  end
  PassFFT{D}(plans, boxes)
end

function apply!(P::PassFFT, x)
  for (ps, bs) in zip(P.plans, P.boxes)
    @maybe_threads length(ps) :dynamic for i in eachindex(ps)
      ps[i] * view(x, bs[i]...)
    end
  end
  x
end
