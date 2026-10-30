# Ablation switches of the CPU NFFT core, for measurements only (not API). Each replaces one
# technique of the default by the alternative another library uses, at the same window, W, σ and
# grid. `ablate!(; kwargs...)` sets them for plans created afterwards (single NFFT plans without
# batch dimensions, `FFTWBackend`); `ablate!()` restores the default. A plan created with a switch
# set holds an `AblatedCore`, which runs its own copies of the resampling loops (below), so the
# default core and its transforms are untouched.
#
# * `fft`: `:passes` (default: pruned per-dimension passes, threaded by NFFT.jl), `:full` (one
#   multidimensional FFTW plan of the whole grid with FFTW's threads), `:unpruned` (per-dimension
#   passes over the whole grid, threaded by NFFT.jl), `:pruned_fftw` (the pruned boxes one after the
#   other, each an FFTW plan with FFTW's threads)
# * `interp`: `:buffer` (default: each tile copied into a private buffer), `:global` (read the
#   padded grid directly)
# * `halo`: `true` (default: halo filled before and folded after the resampling), `false` (no halo
#   fill or fold; the periodic wrap is resolved in the tile copy and in the flush)
# * `flush`: `:tilelocks` (default: one lock per overlapped lock tile), `:globallock` (one SpinLock
#   for all flushes), `:reentrant` (one ReentrantLock for all flushes, taken once per buffer),
#   `:atomic` (no lock, atomic adds)
# * `window`: `:evenodd` (default: one Horner chain in s² for the even and the odd part of the
#   first ⌈W/2⌉ pieces), `:fullhorner` (a Horner chain in s for each of the W pieces, same
#   coefficients)
# * `coords`: `:sorted` (default: node coordinates copied in tile order at plan time),
#   `:permutation` (read from the node matrix through the permutation)
# * `rowdot2`: `true` (default: two FMA chains per row in 2D/3D from W = 8), `false` (one chain)
# * `prefetch`: `true` (default: the value 16 nodes ahead prefetched in the adjoint), `false`
# * `slide`: `true` (default: sliding buffer of the 3D adjoint), `false` (one flush per tile)
# * `schedule`: `:dynamic` (default: each task takes the next tile from a shared counter),
#   `:chunks` (the tiles with dimension 1 fastest, split into one contiguous chunk per task)
# * `bufs`: `:pertask` (default: one private buffer per task), `:pertile` (one buffer per tile,
#   allocated with the plan)
const ABLATION_DEFAULT = (fft=:passes, interp=:buffer, halo=true, flush=:tilelocks, window=:evenodd,
                          coords=:sorted, rowdot2=true, prefetch=true, slide=true, schedule=:dynamic,
                          bufs=:pertask)
const ABLATION = Ref(ABLATION_DEFAULT)

function ablate!(; kwargs...)
  a = merge(ABLATION_DEFAULT, values(kwargs))
  keys(a) == keys(ABLATION_DEFAULT) || throw(ArgumentError("unknown ablation switch in $(keys(kwargs))"))
  a.fft in (:passes, :full, :unpruned, :pruned_fftw) || throw(ArgumentError("fft = $(a.fft)"))
  a.interp in (:buffer, :global) || throw(ArgumentError("interp = $(a.interp)"))
  a.flush in (:tilelocks, :globallock, :reentrant, :atomic) || throw(ArgumentError("flush = $(a.flush)"))
  a.window in (:evenodd, :fullhorner) || throw(ArgumentError("window = $(a.window)"))
  a.coords in (:sorted, :permutation) || throw(ArgumentError("coords = $(a.coords)"))
  a.schedule in (:dynamic, :chunks) || throw(ArgumentError("schedule = $(a.schedule)"))
  a.bufs in (:pertask, :pertile) || throw(ArgumentError("bufs = $(a.bufs)"))
  all(s -> s isa Bool, (a.halo, a.rowdot2, a.prefetch, a.slide)) ||
    throw(ArgumentError("halo, rowdot2, prefetch and slide are Bool"))
  a.interp === :global && !a.halo && throw(ArgumentError("interp = :global needs the halo"))
  ABLATION[] = a
end

# The core with what the switches replace: the kernel (SymKernel or FullKernel), the node
# coordinates in tile order (`binned.xs` or a PermutedNodes), the buffers of the tiles (`nothing`
# for one per task) and the lock of flush = :reentrant.
struct AblatedCore{C<:NFFTCore,S<:NamedTuple,K,X,B}
  core::C
  switches::S
  kernel::K
  xs::X
  tilebufs::B
  lock::ReentrantLock
end

function ablated_core(c::NFFTCore{D,KS,T}, k, a=ABLATION[]) where {D,KS,T}
  c.fft isa FFTWBackend || throw(ArgumentError("the ablation switches need FFTWBackend"))
  if a.fft !== :passes
    g = interior(c.gp, halo(c.W), c.Ñ)
    c = NFFTCore(c.N, c.Ñ, c.W, c.B, c.kernel, c.invphi, c.binned, c.gp,
                 ablated_fft(a.fft, c.fft, g, c.N, Val(:fwd), c.nthreads),
                 ablated_fft(a.fft, c.fft, g, c.N, Val(:bwd), c.nthreads), c.locks, c.bufs, c.fft, c.nthreads)
  end
  K = a.window === :fullhorner ? FullKernel(c.kernel) : c.kernel
  xs = a.coords === :permutation ? PermutedNodes{D}(k, c.binned.perm) : c.binned.xs
  # zeroed once: every trafo overwrites the same part of a tile's buffer
  tb = a.bufs === :pertile ? [zeros(Complex{T}, size(c.bufs[1])) for _ in 1:length(c.binned.tileptr)-1] : nothing
  AblatedCore(c, a, K, xs, tb, ReentrantLock())
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

# The window with each of its W pieces evaluated by its own Horner chain in s (NFFT.jl 0.14's
# evalpoly per piece, FINUFFT, NonuniformFFTs.jl), with the coefficients of the SymKernel: piece
# W−1−i (0-based) at s is piece i at −s.
struct FullKernel{W,P1,T}
  coeffs::NTuple{P1,SVector{W,T}}   # coeffs[k][i]: coefficient of s^(k-1) for support point i
end
function FullKernel(K::SymKernel{W,H,NE,NO,T}) where {W,H,NE,NO,T}
  half(k) = isodd(k) ? K.even[(k+1)÷2] : K.odd[k÷2]
  FullKernel{W,NE + NO,T}(ntuple(k -> SVector{W,T}(ntuple(i -> i <= H ? half(k)[i] : (-1)^(k - 1) * half(k)[W+1-i], Val(W))),
                                 Val(NE + NO)))
end
@generated function weights(K::FullKernel{W,P1,T}, s::T) where {W,P1,T}
  steps = [:(w = muladd.(w, s, c[$k])) for k in P1-1:-1:1]
  quote
    Base.@_inline_meta
    c = K.coeffs
    w = c[$P1]
    $(steps...)
    w
  end
end
@inline function support(K::FullKernel{W,P1,T}, x, Ñ::Int) where {W,P1,T}
  off, s = support_offset(x, Ñ, W, T)
  off, weights(K, s)
end

# The node coordinates in tile order, read from the node matrix k through the permutation.
struct PermutedNodes{D,T,M<:AbstractMatrix{T}} <: AbstractVector{NTuple{D,T}}
  k::M
  perm::Vector{Int}
end
PermutedNodes{D}(k::AbstractMatrix{T}, perm) where {D,T} = PermutedNodes{D,T,typeof(k)}(k, perm)
Base.size(x::PermutedNodes) = size(x.perm)
@inline function Base.getindex(x::PermutedNodes{D}, q::Int) where {D}
  @boundscheck checkbounds(x, q)
  @inbounds j = x.perm[q]
  ntuple(d -> @inbounds(x.k[d, j]), Val(D))
end

# One row chain instead of rowdot2 (the generic method of resample_node).
@inline resample_node1(g::AbstractArray{C,D}, off::NTuple{D,Int}, ws::NTuple{D,SVector{W,T}}) where {C,D,W,T} =
  invoke(resample_node, Tuple{AbstractArray{C,D},NTuple{D,Int},NTuple{D,SVector{W,T}}}, g, off, ws)

# f(it, order[i]) for i = 1 … length(order) on ntasks tasks, after init(it) on each task.
# :dynamic: the next index from a shared counter; :chunks: one contiguous chunk per task (NFFT.jl
# 0.14: OhMyThreads' DynamicScheduler with nthreads chunks).
function foreach_tile(f, init, order, ntasks, ::Val{sched}) where {sched}
  n = length(order)
  if sched === :chunks
    cs = chunks(1:n, ntasks)
    tasks(length(cs)) do it
      init(it)
      for i in cs[it]
        f(it, order[i])
      end
    end
  else
    next = Threads.Atomic{Int}(1)
    tasks(min(ntasks, n)) do it
      init(it)
      while (i = Threads.atomic_add!(next, 1)) <= n
        f(it, order[i])
      end
    end
  end
  nothing
end

# The buffer of task it for tile t.
@inline tilebuf(bufs, ::Nothing, it, t) = bufs[it]
@inline tilebuf(bufs, tb, it, t) = tb[t]

function trafo_core!(out, a::AblatedCore, f)
  c = a.core; s = a.switches
  h = halo(c.W); g = interior(c.gp, h, c.Ñ)
  @stage :zero threaded_fill!(c.gp, 0, c.nthreads)
  @stage :correction resampling_correction_core!(g, f, c.invphi, c.N, c.Ñ, c.nthreads)
  @stage :fft apply!(c.fwd, g)
  if s.halo
    @stage :halo fill_halo!(c.gp, h, c.nthreads; Ñs=c.Ñ)
  else
    STAGE_TIMES[findfirst(==(:halo), STAGES)] = 0
  end
  @stage :resampling begin
    sched = Val(s.schedule)
    node = s.rowdot2 ? resample_node : resample_node1
    if s.interp === :global
      ab_resample_global!(out, c.gp, a.kernel, a.xs, c.binned, c.Ñ, length(c.bufs), node, sched)
    else
      cp = s.halo ? tile_copy! : WrapCopy(h[1], c.Ñ)
      ab_resample!(out, c.gp, a.kernel, a.xs, c.binned, c.Ñ, c.bufs, a.tilebufs, cp, node, sched)
    end
  end
  out
end

function adjoint_core!(f, a::AblatedCore, vals)
  c = a.core; s = a.switches
  h = halo(c.W); g = interior(c.gp, h, c.Ñ)
  @stage :zero threaded_fill!(c.gp, 0, c.nthreads)
  @stage :resampling begin
    fl = s.flush === :globallock ? flush_global! : s.flush === :atomic ? flush_atomic! :
         s.flush === :reentrant ? flush_nolock! : flush!
    s.halo || (fl = WrapFlush(fl, h[1], c.Ñ))
    s.flush === :reentrant && (fl = LockedFlush(a.lock, fl))
    sched = Val(s.schedule); pf = Val(s.prefetch)
    if length(c.N) >= 3 && s.slide
      ab_resample_adjoint_slide!(c.gp, a.kernel, a.xs, c.binned, vals, c.Ñ, c.locks, c.bufs, a.tilebufs, fl, pf, sched)
    else
      ab_resample_adjoint_tiles!(c.gp, a.kernel, a.xs, c.binned, vals, c.Ñ, c.locks, c.bufs, a.tilebufs, fl, pf, sched)
    end
  end
  if s.halo
    @stage :halo fold_halo!(c.gp, h, c.nthreads; Ñs=c.Ñ)
  else
    STAGE_TIMES[findfirst(==(:halo), STAGES)] = 0
  end
  @stage :fft apply!(c.bwd, g)
  @stage :correction resampling_correction_adjoint_core!(f, g, c.invphi, c.N, c.Ñ, c.nthreads)
  f
end

# resample! of resampling.jl with the kernel K, the coordinates xs, the copy cp, the node function
# and the schedule as arguments.
function ab_resample!(out, gp, K, xs, b::Binned{D}, Ñ, bufs, tb, cp, node::F, sched) where {D,F}
  foreach_tile(it -> tb === nothing && fill!(bufs[it], 0), b.order1, length(bufs), sched) do it, t
    r = b.tileptr[t]:b.tileptr[t+1]-1
    isempty(r) && return
    buf = tilebuf(bufs, tb, it, t)
    o = b.origins[t]
    cp(buf, gp, o)
    @inbounds for q in r
      sw = map(@inline((x, n) -> support(K, x, n)), xs[q], Ñ)
      out[b.perm[q]] = node(buf, map(first, sw) .- o, map(last, sw))
    end
  end
  out
end

# Interpolation from the padded grid itself, tiles taken as in `ab_resample!` (FINUFFT and NFFT3
# read the global grid; here without index wrapping, thanks to the halo).
function ab_resample_global!(out, gp, K, xs, b::Binned{D}, Ñ, ntasks, node::F, sched) where {D,F}
  foreach_tile(_ -> nothing, b.order1, ntasks, sched) do it, t
    @inbounds for q in b.tileptr[t]:b.tileptr[t+1]-1
      sw = map(@inline((x, n) -> support(K, x, n)), xs[q], Ñ)
      out[b.perm[q]] = node(gp, map(first, sw), map(last, sw))
    end
  end
  out
end

# spread_nodes! of resampling.jl with the kernel K, the coordinates xs and the prefetch switchable.
@inline function ab_spread_nodes!(buf, K, xs, b::Binned{D}, vals, Ñ, o, r, ::Val{pf}) where {D,pf}
  @inbounds for q in r
    pf && q + 16 <= last(r) && prefetch(vals, b.perm[q+16])
    sw = map(@inline((x, m) -> support(K, x, m)), xs[q], Ñ)
    resample_adjoint_node!(buf, map(first, sw) .- o, map(last, sw), vals[b.perm[q]])
  end
  buf
end

# resample_adjoint_tiles! of resampling.jl; with :chunks the tiles in the order of the resampling
# (dimension 1 fastest, as NFFT.jl 0.14), otherwise in tile order as the default.
function ab_resample_adjoint_tiles!(gp, K, xs, b::Binned{D}, vals, Ñ, L, bufs, tb, fl::F, pf, sched) where {D,F}
  order = sched === Val(:chunks) ? b.order1 : (1:length(b.tileptr)-1)
  foreach_tile(_ -> nothing, order, length(bufs), sched) do it, t
    r = b.tileptr[t]:b.tileptr[t+1]-1
    isempty(r) && return
    buf = tilebuf(bufs, tb, it, t)
    o = b.origins[t]
    fill!(buf, 0)
    ab_spread_nodes!(buf, K, xs, b, vals, Ñ, o, r, pf)
    fl(gp, buf, o, L)
  end
  gp
end

# resample_adjoint_slide! of resampling.jl (per-tile buffers: that of the first tile of a segment).
function ab_resample_adjoint_slide!(gp, K, xs, b::Binned{D}, vals, Ñ, L, bufs, tb, fl::F, pf, sched) where {D,F}
  ntD = b.ntiles[D]; ncol = prod(Base.front(b.ntiles))
  ntasks = length(bufs); B = L.B
  nseg = clamp(cld(8ntasks, ncol), 1, ntD); seglen = cld(ntD, nseg)
  segs = [(c - 1) * ntD .+ (s:min(s + seglen - 1, ntD)) for c in 1:ncol for s in 1:seglen:ntD]
  E = size(bufs[1], 1)
  S = E^(D - 1)
  foreach_tile(_ -> nothing, segs, ntasks, sched) do it, tiles
    b.tileptr[last(tiles)+1] > b.tileptr[first(tiles)] || return
    buf = tilebuf(bufs, tb, it, first(tiles))
    fill!(buf, 0)
    for t in tiles
      o = b.origins[t]
      ab_spread_nodes!(buf, K, xs, b, vals, Ñ, o, b.tileptr[t]:b.tileptr[t+1]-1, pf)
      if t == last(tiles)
        fl(gp, buf, o, L)
      else
        fl(gp, lastdim_view(buf, 1:B), o, L)
        @inbounds for i in 1:(E-B)*S
          buf[i] = buf[i+B*S]
        end
        @inbounds for i in (E-B)*S+1:E*S
          buf[i] = zero(eltype(buf))
        end
      end
    end
  end
  gp
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

# The flush without a lock, for the global locks below.
function flush_nolock!(gp, buf, o::NTuple{D,Int}, L) where {D}
  lo = o .+ 1
  hi = min.(lo .+ size(buf) .- 1, size(gp))
  @inbounds for Jt in CartesianIndices(ntuple(d -> lo[d+1]:hi[d+1], Val(D - 1)))
    jt = Tuple(Jt); bt = jt .- Base.tail(o)
    @simd for j1 in lo[1]:hi[1]
      gp[j1, jt...] += buf[j1 - o[1], bt...]
    end
  end
end

# One lock for all flushes (NFFT.jl 0.14 and NonuniformFFTs.jl, there a ReentrantLock; FINUFFT 2.5.1
# an OpenMP critical section): the same SpinLock as the default, taken once per flush (per piece
# without the halo).
function flush_global!(gp, buf, o::NTuple{D,Int}, L::TileLocks{D}) where {D}
  l = L.locks[1]
  lock(l)
  flush_nolock!(gp, buf, o, L)
  unlock(l)
end

# A flush under one ReentrantLock, taken once per buffer as in NFFT.jl 0.14's addBlock!.
struct LockedFlush{F}
  lock::ReentrantLock
  fl::F
end
(l::LockedFlush)(gp, buf, o, L) = lock(() -> l.fl(gp, buf, o, L), l.lock)

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
