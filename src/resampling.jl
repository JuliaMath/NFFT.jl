# Resampling (equidistant grid → nodes) and adjoint resampling (nodes → grid) on the CPU, on the
# padded grid of geometry.jl. Loops run over the support rows: an explicit inner loop along the
# contiguous first dimension, weights of the other dimensions multiplied once per row (as fast as
# Base.Cartesian code, and generic in D).

@inline function resample_node(g::AbstractArray{C,D}, off::NTuple{D,Int}, ws::NTuple{D,SVector{W,T}}) where {C,D,W,T}
  s = zero(C)
  @inbounds for It in CartesianIndices(ntuple(_ -> Base.OneTo(W), Val(D - 1)))
    wt = prod(ntuple(d -> ws[d+1][It[d]], Val(D - 1)); init=one(T))
    jt = Base.tail(off) .+ Tuple(It)
    acc = zero(C)
    @simd for i in 1:W
      # @inbounds on the statement itself: on the outer loop it did not reach into @simd
      @inbounds acc += ws[1][i] * g[off[1]+i, jt...]
    end
    s += wt * acc
  end
  s
end

# w₁·v is formed once per node, so every grid point of a row takes one FMA.
@inline function resample_adjoint_node!(g::AbstractArray{C,D}, off::NTuple{D,Int}, ws::NTuple{D,SVector{W,T}}, v::C) where {C,D,W,T}
  w1v = ws[1] .* v
  @inbounds for It in CartesianIndices(ntuple(_ -> Base.OneTo(W), Val(D - 1)))
    wt = prod(ntuple(d -> ws[d+1][It[d]], Val(D - 1)); init=one(T))
    jt = Base.tail(off) .+ Tuple(It)
    for i in 1:W
      g[off[1]+i, jt...] = muladd(wt, w1v[i], g[off[1]+i, jt...])
    end
  end
  g
end

# Resampling from a private copy of each tile with its halo: reading the global grid directly
# made the time depend on the grid strides through cache set conflicts (3D, W = 10: 8.3–13.7 ms
# for Ñ = 96…128); the compact buffer removes that dependence.
function resample!(out, gp, K::SymKernel{W}, b::Binned{D}, Ñ, bufs) where {W,D}
  ntile = length(b.tileptr) - 1
  next = Threads.Atomic{Int}(1)
  @sync for it in 1:min(length(bufs), ntile)
    Threads.@spawn begin
      buf = bufs[it]
      fill!(buf, 0)
      while (i = Threads.atomic_add!(next, 1)) <= ntile
        t = b.order1[i]
        r = b.tileptr[t]:b.tileptr[t+1]-1
        isempty(r) && continue
        o = b.origins[t]
        hi = min.(o .+ size(buf), size(gp))
        src = view(gp, map((a, z) -> a+1:z, o, hi)...)
        copyto!(view(buf, map(s -> 1:s, size(src))...), src)
        @inbounds for q in r
          sw = map(@inline((x, n) -> support(K, x, n)), b.xs[q], Ñ)
          out[b.perm[q]] = resample_node(buf, map(first, sw) .- o, map(last, sw))
        end
      end
    end
  end
  out
end

# Add a private buffer (padded indices o+1 … o+size(buf)) to the grid, locking each overlapped lock
# tile separately (one lock at a time, so no ordering problem). Explicit SIMD loop along the first
# dimension: a CartesianIndices loop over the box did not vectorise.
function flush!(gp, buf, o::NTuple{D,Int}, L::TileLocks{D}) where {D}
  B = L.B
  lo = o .+ 1
  hi = min.(lo .+ size(buf) .- 1, size(gp))
  for I in CartesianIndices(ntuple(d -> fld(lo[d] - 1, B):fld(hi[d] - 1, B), D))
    r = ntuple(d -> max(lo[d], I[d] * B + 1):min(hi[d], (I[d] + 1) * B), D)
    lock(L.locks[(Tuple(I) .+ 1)...])
    @inbounds for Jt in CartesianIndices(Base.tail(r))
      jt = Tuple(Jt); bt = jt .- Base.tail(o)
      @simd for j1 in r[1]
        gp[j1, jt...] += buf[j1 - o[1], bt...]
      end
    end
    unlock(L.locks[(Tuple(I) .+ 1)...])
  end
end

# Adjoint resampling: each task takes the next tile, spreads it into a private zeroed buffer of edge
# B+W+1 (it stays in L1/L2) and flushes it into the grid (ducc's scheme; checkerboard colouring of
# tiles without buffers was slower because neighbouring tiles reload shared halo rows).
function resample_adjoint_tiles!(gp, K::SymKernel{W}, b::Binned{D}, vals, Ñ, L::TileLocks{D}, bufs) where {W,D}
  ntile = length(b.tileptr) - 1
  next = Threads.Atomic{Int}(1)
  @sync for it in 1:min(length(bufs), ntile)
    Threads.@spawn begin
      buf = bufs[it]
      while (t = Threads.atomic_add!(next, 1)) <= ntile
        r = b.tileptr[t]:b.tileptr[t+1]-1
        isempty(r) && continue
        o = b.origins[t]
        fill!(buf, 0)
        @inbounds for q in r
          sw = map(@inline((x, n) -> support(K, x, n)), b.xs[q], Ñ)
          resample_adjoint_node!(buf, map(first, sw) .- o, map(last, sw), vals[b.perm[q]])
        end
        flush!(gp, buf, o, L)
      end
    end
  end
  gp
end

# 3D: the buffer slides along dimension D over a segment of a tile column (ducc's shift-and-flush):
# after each tile only the B slices the next tile no longer touches are flushed, the remaining W+1
# slices move to the front. Every grid point is flushed from (E/B)^(D−1) instead of (E/B)^D
# buffers, E = B+W+1 (3D, W = 13: 37.6 → 29.4 ms).
function resample_adjoint_slide!(gp, K::SymKernel{W}, b::Binned{D}, vals, Ñ, L::TileLocks{D}, bufs) where {W,D}
  ntD = b.ntiles[D]; ncol = prod(Base.front(b.ntiles))
  ntasks = length(bufs); B = L.B
  nseg = clamp(cld(2ntasks, ncol), 1, ntD); seglen = cld(ntD, nseg)
  segs = [(c - 1) * ntD .+ (s:min(s + seglen - 1, ntD)) for c in 1:ncol for s in 1:seglen:ntD]
  E = size(bufs[1], 1)
  next = Threads.Atomic{Int}(1)
  @sync for it in 1:min(ntasks, length(segs))
    Threads.@spawn begin
      buf = bufs[it]
      S = E^(D - 1)
      while (si = Threads.atomic_add!(next, 1)) <= length(segs)
        tiles = segs[si]
        b.tileptr[last(tiles)+1] > b.tileptr[first(tiles)] || continue
        fill!(buf, 0)
        for t in tiles
          o = b.origins[t]
          @inbounds for q in b.tileptr[t]:b.tileptr[t+1]-1
            sw = map(@inline((x, n) -> support(K, x, n)), b.xs[q], Ñ)
            resample_adjoint_node!(buf, map(first, sw) .- o, map(last, sw), vals[b.perm[q]])
          end
          if t == last(tiles)
            flush!(gp, buf, o, L)
          else
            flush!(gp, selectdim(buf, D, 1:B), o, L)
            # forward copy is safe: the destination starts before the source
            @inbounds for i in 1:(E-B)*S
              buf[i] = buf[i+B*S]
            end
            @inbounds for i in (E-B)*S+1:E*S
              buf[i] = zero(eltype(buf))
            end
          end
        end
      end
    end
  end
  gp
end

resample_adjoint!(gp, K, b::Binned{D}, vals, Ñ, L, bufs) where {D} =
  D >= 3 ? resample_adjoint_slide!(gp, K, b, vals, Ñ, L, bufs) :
           resample_adjoint_tiles!(gp, K, b, vals, Ñ, L, bufs)

# Halo of the padded grid, h = halo(W): fill it periodically from the interior before resampling,
# fold it back onto the interior after the adjoint resampling. One dimension after the other (this
# covers edges and corners), threaded over another dimension.
function halo_chunks(gp, d, nthreads)
  e = d == ndims(gp) ? 1 : ndims(gp)
  e, chunks(axes(gp, e), nthreads)
end
function fill_halo!(gp, h::Tuple, nthreads)
  for d in 1:ndims(gp)
    Ñ = size(gp, d) - h[1] - h[2]
    if ndims(gp) == 1
      copyto!(view(gp, 1:h[1]), view(gp, Ñ+1:Ñ+h[1]))
      copyto!(view(gp, Ñ+h[1]+1:Ñ+h[1]+h[2]), view(gp, h[1]+1:h[1]+h[2]))
      continue
    end
    e, cs = halo_chunks(gp, d, nthreads)
    Threads.@threads for c in cs
      g = selectdim(gp, e, c)
      copyto!(selectdim(g, d, 1:h[1]), selectdim(g, d, Ñ+1:Ñ+h[1]))
      copyto!(selectdim(g, d, Ñ+h[1]+1:Ñ+h[1]+h[2]), selectdim(g, d, h[1]+1:h[1]+h[2]))
    end
  end
  gp
end
function fold_halo!(gp, h::Tuple, nthreads)
  for d in ndims(gp):-1:1
    Ñ = size(gp, d) - h[1] - h[2]
    if ndims(gp) == 1
      view(gp, Ñ+1:Ñ+h[1]) .+= view(gp, 1:h[1])
      view(gp, h[1]+1:h[1]+h[2]) .+= view(gp, Ñ+h[1]+1:Ñ+h[1]+h[2])
      continue
    end
    e, cs = halo_chunks(gp, d, nthreads)
    Threads.@threads for c in cs
      g = selectdim(gp, e, c)
      selectdim(g, d, Ñ+1:Ñ+h[1]) .+= selectdim(g, d, 1:h[1])
      selectdim(g, d, h[1]+1:h[1]+h[2]) .+= selectdim(g, d, Ñ+h[1]+1:Ñ+h[1]+h[2])
    end
  end
  gp
end
interior(gp, h::Tuple) = view(gp, map(n -> h[1]+1:n-h[2], size(gp))...)
