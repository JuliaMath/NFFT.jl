# Ablation switches of the CPU NFFT core, for measurements only (not API). Each replaces one
# technique of the default by the alternative another library uses, at the same window, W, σ and
# grid. `ablate!(; kwargs...)` sets them for plans created afterwards (single NFFT plans without
# batch dimensions, `FFTWBackend`); `ablate!()` restores the default. A plan created with a switch
# set holds an `AblatedCore`, so the default core and its transforms are untouched.
#
# * `fft`: `:passes` (default: pruned per-dimension passes, threaded by NFFT.jl), `:full` (one
#   multidimensional FFTW plan of the whole grid with FFTW's threads), `:unpruned` (per-dimension
#   passes over the whole grid, threaded by NFFT.jl), `:pruned_fftw` (the pruned boxes one after the
#   other, each an FFTW plan with FFTW's threads)
# * `interp`: `:buffer` (default: each tile copied into a private buffer), `:global` (read the
#   padded grid directly)
# * `halo`: `true` (default: halo filled before and folded after the resampling), `false` (no halo
#   fill or fold; the periodic wrap is resolved in the tile copy and in the flush)
# * `flush`: `:tilelocks` (default: one lock per overlapped lock tile), `:globallock` (one lock for
#   all flushes), `:atomic` (no lock, atomic adds)
const ABLATION_DEFAULT = (fft=:passes, interp=:buffer, halo=true, flush=:tilelocks)
const ABLATION = Ref(ABLATION_DEFAULT)

function ablate!(; kwargs...)
  a = merge(ABLATION_DEFAULT, values(kwargs))
  keys(a) == keys(ABLATION_DEFAULT) || throw(ArgumentError("unknown ablation switch in $(keys(kwargs))"))
  a.fft in (:passes, :full, :unpruned, :pruned_fftw) || throw(ArgumentError("fft = $(a.fft)"))
  a.interp in (:buffer, :global) || throw(ArgumentError("interp = $(a.interp)"))
  a.flush in (:tilelocks, :globallock, :atomic) || throw(ArgumentError("flush = $(a.flush)"))
  a.interp === :global && !a.halo && throw(ArgumentError("interp = :global needs the halo"))
  ABLATION[] = a
end

struct AblatedCore{C<:NFFTCore}
  core::C
  interp::Symbol
  halo::Bool
  flush::Symbol
end

function ablated_core(c::NFFTCore, a=ABLATION[])
  c.fft isa FFTWBackend || throw(ArgumentError("the ablation switches need FFTWBackend"))
  if a.fft !== :passes
    g = interior(c.gp, halo(c.W))
    c = NFFTCore(c.N, c.Ñ, c.W, c.B, c.kernel, c.invphi, c.binned, c.gp,
                 ablated_fft(a.fft, c.fft, g, c.N, Val(:fwd), c.nthreads),
                 ablated_fft(a.fft, c.fft, g, c.N, Val(:bwd), c.nthreads), c.locks, c.bufs, c.fft, c.nthreads)
  end
  AblatedCore(c, a.interp, a.halo, a.flush)
end

slice_size(a::AblatedCore) = a.core.N

# The FFT variants as a PassFFT: a list of passes, each a list of (plan, box) that `apply!` runs in
# parallel on NFFT.jl's tasks. :full is one pass of one plan, :pruned_fftw one pass per box.
function ablated_fft(mode, fft, x::AbstractArray{C,D}, N, ::Val{dir}, nthreads) where {C,D,dir}
  K = dir === :fwd ? Val(-1) : Val(1)
  Ñ = size(x)
  full = ntuple(k -> 1:Ñ[k], D)
  if mode === :full
    p = planning(() -> pass_plan(fft, K, view(x, full...), 1:D), fft, nthreads)
    return PassFFT([[p]], [[full]])
  end
  order = dir === :fwd ? reverse(1:D) : 1:D
  if mode === :unpruned
    # the boxes of pass_boxes without pruning: whole grid, split for threading along the outermost
    # dimension other than d (1D: one plan threaded by FFTW, as the default)
    boxes = map(order) do d
      D == 1 && return [full]
      s = d == D ? D - 1 : D
      [Base.setindex(full, c, s) for c in chunks(full[s], nthreads)]
    end
    plans = planning(fft, D == 1 ? fft_nthreads(length(x), nthreads) : 1) do
      [pass_plans(fft, K, [view(x, b...) for b in bs], d) for (d, bs) in zip(order, boxes)]
    end
    return PassFFT(plans, boxes)
  end
  # :pruned_fftw: the pruned boxes, not split, each its own pass
  bs = [(d, b) for d in order for b in pass_boxes(x, N, d, 1:D, true, nthreads)]
  plans = planning(fft, nthreads) do
    [[pass_plan(fft, K, view(x, b...), d)] for (d, b) in bs]
  end
  PassFFT(plans, [[b] for (_, b) in bs])
end

function trafo_core!(out, a::AblatedCore, f)
  c = a.core
  h = halo(c.W); g = interior(c.gp, h)
  @stage :zero threaded_fill!(c.gp, 0, c.nthreads)
  @stage :correction resampling_correction_core!(g, f, c.invphi, c.N, c.Ñ, c.nthreads)
  @stage :fft apply!(c.fwd, g)
  if a.halo
    @stage :halo fill_halo!(c.gp, h, c.nthreads)
  else
    STAGE_TIMES[findfirst(==(:halo), STAGES)] = 0
  end
  @stage :resampling begin
    if a.interp === :global
      resample_global!(out, c.gp, c.kernel, c.binned, c.Ñ, length(c.bufs))
    elseif a.halo
      resample!(out, c.gp, c.kernel, c.binned, c.Ñ, c.bufs)
    else
      resample!(out, c.gp, c.kernel, c.binned, c.Ñ, c.bufs, WrapCopy(h[1], c.Ñ))
    end
  end
  out
end

function adjoint_core!(f, a::AblatedCore, vals)
  c = a.core
  h = halo(c.W); g = interior(c.gp, h)
  @stage :zero threaded_fill!(c.gp, 0, c.nthreads)
  @stage :resampling begin
    fl = a.flush === :globallock ? flush_global! : a.flush === :atomic ? flush_atomic! : flush!
    if a.halo
      resample_adjoint!(c.gp, c.kernel, c.binned, vals, c.Ñ, c.locks, c.bufs, fl)
    else
      resample_adjoint!(c.gp, c.kernel, c.binned, vals, c.Ñ, c.locks, c.bufs, WrapFlush(fl, h[1], c.Ñ))
    end
  end
  if a.halo
    @stage :halo fold_halo!(c.gp, h, c.nthreads)
  else
    STAGE_TIMES[findfirst(==(:halo), STAGES)] = 0
  end
  @stage :fft apply!(c.bwd, g)
  @stage :correction resampling_correction_adjoint_core!(f, g, c.invphi, c.N, c.Ñ, c.nthreads)
  f
end

# Interpolation from the padded grid itself, tiles taken as in `resample!` (FINUFFT and NFFT3 read
# the global grid; here without index wrapping, thanks to the halo).
function resample_global!(out, gp, K::SymKernel{W}, b::Binned{D}, Ñ, ntasks) where {W,D}
  ntile = length(b.tileptr) - 1
  next = Threads.Atomic{Int}(1)
  tasks(min(ntasks, ntile)) do _
    while (i = Threads.atomic_add!(next, 1)) <= ntile
      t = b.order1[i]
      @inbounds for q in b.tileptr[t]:b.tileptr[t+1]-1
        sw = map(@inline((x, n) -> support(K, x, n)), b.xs[q], Ñ)
        out[b.perm[q]] = resample_node(gp, map(first, sw), map(last, sw))
      end
    end
  end
  out
end

# Without a halo: padded index p of a tile buffer stands for interior index mod(p − h1 − 1, Ñ) + 1.
# The (at most two) contiguous pieces of the padded range p0+1 … p0+n in one dimension, as pairs
# (buffer range, padded range of the interior point it maps to). Needs n ≤ Ñ.
@inline function wrap_pieces(p0, n, h1, Ñ)
  n <= Ñ || throw(ArgumentError("tile buffer larger than the grid"))
  j = mod(p0 - h1, Ñ) + 1
  l = min(n, Ñ - j + 1)
  ((1:l, h1+j:h1+j+l-1), (l+1:n, h1+1:h1+n-l))
end

# The boxes of a buffer of size sz at origin o (truncated to the padded grid as in the default)
# with their wrapped destinations: f(buffer ranges, padded ranges) for each nonempty one.
function foreach_wrapped(f, o::NTuple{D,Int}, sz, gsz, h1, Ñ) where {D}
  n = min.(sz, gsz .- o)
  ps = ntuple(d -> wrap_pieces(o[d], n[d], h1, Ñ[d]), Val(D))
  for I in CartesianIndices(ntuple(_ -> 2, Val(D)))
    br = ntuple(d -> ps[d][I[d]][1], Val(D))
    any(isempty, br) && continue
    f(br, ntuple(d -> ps[d][I[d]][2], Val(D)))
  end
end

# Tile copy with the wrap (ducc, NFFT.jl 0.14).
struct WrapCopy{D}
  h1::Int
  Ñ::NTuple{D,Int}
end
function (w::WrapCopy)(buf, gp, o)
  foreach_wrapped(o, size(buf), size(gp), w.h1, w.Ñ) do br, pr
    copyto!(view(buf, br...), view(gp, pr...))
  end
end

# Flush with the wrap: each piece through the flush `fl` at the origin of its destination.
struct WrapFlush{F,D}
  fl::F
  h1::Int
  Ñ::NTuple{D,Int}
end
function (w::WrapFlush)(gp, buf, o, L)
  foreach_wrapped(o, size(buf), size(gp), w.h1, w.Ñ) do br, pr
    w.fl(gp, view(buf, br...), map(r -> first(r) - 1, pr), L)
  end
end

# One lock for all flushes (NFFT.jl 0.14 and NonuniformFFTs.jl, there a ReentrantLock; FINUFFT 2.5.1
# an OpenMP critical section): the same SpinLock as the default, taken once per flush.
function flush_global!(gp, buf, o::NTuple{D,Int}, L::TileLocks{D}) where {D}
  lo = o .+ 1
  hi = min.(lo .+ size(buf) .- 1, size(gp))
  l = L.locks[1]
  lock(l)
  @inbounds for Jt in CartesianIndices(ntuple(d -> lo[d+1]:hi[d+1], Val(D - 1)))
    jt = Tuple(Jt); bt = jt .- Base.tail(o)
    @simd for j1 in lo[1]:hi[1]
      gp[j1, jt...] += buf[j1 - o[1], bt...]
    end
  end
  unlock(l)
end

# No lock, an atomic add per real and imaginary part (FINUFFT 2.6-dev above 10 threads).
function flush_atomic!(gp::Array{Complex{T},D}, buf, o::NTuple{D,Int}, L) where {T,D}
  lo = o .+ 1
  hi = min.(lo .+ size(buf) .- 1, size(gp))
  GC.@preserve gp begin
    @inbounds for Jt in CartesianIndices(ntuple(d -> lo[d+1]:hi[d+1], Val(D - 1)))
      jt = Tuple(Jt); bt = jt .- Base.tail(o)
      for j1 in lo[1]:hi[1]
        v = buf[j1 - o[1], bt...]
        p = Ptr{T}(pointer(gp, LinearIndices(gp)[j1, jt...]))
        Base.unsafe_modify!(p, +, real(v), :monotonic)
        Base.unsafe_modify!(p + sizeof(T), +, imag(v), :monotonic)
      end
    end
  end
end
