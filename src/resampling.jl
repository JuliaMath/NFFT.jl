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
      # @inbounds on the statement itself: on the outer loop it does not reach into @simd
      @inbounds acc += ws[1][i] * g[off[1]+i, jt...]
    end
    s += wt * acc
  end
  s
end

# On the CPU in 2D/3D from W = 8: each row summed in two independent chains. The loop above runs all
# W FMAs of a row into one accumulator (no reassociation for complex sums), a chain of W · 4 cycles;
# two chains are 4–14 % faster for W = 10…14 and 1.2–1.7× for W = 16, a little slower for W = 6
# and, in the full 1D transform, 12 % slower for W = 15.
@inline function resample_node(g::Array{C,D}, off::NTuple{D,Int}, ws::NTuple{D,SVector{W,T}}) where {C,D,W,T}
  (D == 1 || W < 8) && return invoke(resample_node, Tuple{AbstractArray{C,D},NTuple{D,Int},NTuple{D,SVector{W,T}}}, g, off, ws)
  s = zero(C)
  @inbounds for It in CartesianIndices(ntuple(_ -> Base.OneTo(W), Val(D - 1)))
    wt = prod(ntuple(d -> ws[d+1][It[d]], Val(D - 1)); init=one(T))
    s = muladd(wt, rowdot2(g, off[1], Base.tail(off) .+ Tuple(It), ws[1]), s)
  end
  s
end

# Σᵢ w[i] g[o1+i, jt...], odd and even i in separate chains.
@generated function rowdot2(g, o1, jt, w::SVector{W}) where {W}
  ex = [:(a1 = w[1] * g[o1+1, jt...]), :(a2 = w[2] * g[o1+2, jt...])]
  for i in 3:W
    a = isodd(i) ? :a1 : :a2
    push!(ex, :($a = muladd(w[$i], g[o1+$i, jt...], $a)))
  end
  quote
    Base.@_inline_meta
    @inbounds begin $(ex...) end
    a1 + a2
  end
end

# w₁·v is formed once per node, so every grid point of a row takes one FMA.
@inline function resample_adjoint_node!(g::AbstractArray{C,D}, off::NTuple{D,Int}, ws::NTuple{D,SVector{W,T}}, v) where {C,D,W,T}
  w1v = ws[1] .* convert(C, v)
  @inbounds for It in CartesianIndices(ntuple(_ -> Base.OneTo(W), Val(D - 1)))
    wt = prod(ntuple(d -> ws[d+1][It[d]], Val(D - 1)); init=one(T))
    jt = Base.tail(off) .+ Tuple(It)
    for i in 1:W
      g[off[1]+i, jt...] = muladd(wt, w1v[i], g[off[1]+i, jt...])
    end
  end
  g
end

# Resampling from a private copy of each tile with its halo: read directly from the global grid,
# the time depends on the grid strides through cache set conflicts (3D, W = 10: 8.3–13.7 ms for
# Ñ = 96…128); from the compact buffer it does not.
function resample!(out, gp, K::SymKernel{W}, b::Binned{D}, Ñ, bufs, cp=tile_copy!) where {W,D}
  ntile = length(b.tileptr) - 1
  next = Threads.Atomic{Int}(1)
  tasks(min(length(bufs), ntile)) do it
    buf = bufs[it]
    fill!(buf, 0)
    while (i = Threads.atomic_add!(next, 1)) <= ntile
      t = b.order1[i]
      r = b.tileptr[t]:b.tileptr[t+1]-1
      isempty(r) && continue
      o = b.origins[t]
      cp(buf, gp, o)
      @inbounds for q in r
        sw = map(@inline((x, n) -> support(K, x, n)), b.xs[q], Ñ)
        out[b.perm[q]] = resample_node(buf, map(first, sw) .- o, map(last, sw))
      end
    end
  end
  out
end

# Copy the tile at origin o with its halo into buf (the part inside the padded grid).
function tile_copy!(buf, gp, o)
  hi = min.(o .+ size(buf), size(gp))
  src = view(gp, map((a, z) -> a+1:z, o, hi)...)
  copyto!(view(buf, map(s -> 1:s, size(src))...), src)
end

# Add a private buffer (padded indices o+1 … o+size(buf)) to the grid, locking each overlapped lock
# tile separately (one lock at a time, so no ordering problem). Explicit SIMD loop along the first
# dimension: a CartesianIndices loop over the box does not vectorise.
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

# Spread the sorted nodes q ∈ r into the buffer of a tile with origin o. vals[perm[q]] is read in
# random order and misses the cache; behind the stores of one node the load for the next issues too
# late to overlap, so the value 16 nodes ahead is prefetched (2D, W = 10: a quarter more time
# without).
@inline function spread_nodes!(buf, K::SymKernel{W}, b::Binned{D}, vals, Ñ, o, r) where {W,D}
  @inbounds for q in r
    q + 16 <= last(r) && prefetch(vals, b.perm[q+16])
    sw = map(@inline((x, m) -> support(K, x, m)), b.xs[q], Ñ)
    resample_adjoint_node!(buf, map(first, sw) .- o, map(last, sw), vals[b.perm[q]])
  end
  buf
end

# Hint to load a[i] into the cache (read, high locality); no effect on the result.
# Julia < 1.12 rejects the intrinsic in ccall(…, llvmcall, …) and needs it as typed-pointer IR.
@static if VERSION >= v"1.12"
  @inline prefetch(a::StridedArray, i) =
    ccall("llvm.prefetch", llvmcall, Cvoid, (Ptr{Cvoid}, Int32, Int32, Int32), pointer(a, i), 0, 3, 1)
else
  @inline prefetch(a::StridedArray, i) = Base.llvmcall(("""
    declare void @llvm.prefetch(i8*, i32, i32, i32)
    define void @entry(i64 %p) {
      %ptr = inttoptr i64 %p to i8*
      call void @llvm.prefetch(i8* %ptr, i32 0, i32 3, i32 1)
      ret void
    }""", "entry"), Cvoid, Tuple{Int}, Int(pointer(a, i)))
end
@inline prefetch(a, i) = nothing

# Adjoint resampling: each task takes the next tile, adds its nodes into a private zeroed buffer of
# edge B+W+1 (it stays in L1/L2) and flushes it into the grid (ducc's scheme; checkerboard colouring
# of tiles without buffers is slower because neighbouring tiles reload shared halo rows).
function resample_adjoint_tiles!(gp, K::SymKernel{W}, b::Binned{D}, vals, Ñ, L::TileLocks{D}, bufs, fl=flush!) where {W,D}
  ntile = length(b.tileptr) - 1
  next = Threads.Atomic{Int}(1)
  tasks(min(length(bufs), ntile)) do it
    buf = bufs[it]
    while (t = Threads.atomic_add!(next, 1)) <= ntile
      r = b.tileptr[t]:b.tileptr[t+1]-1
      isempty(r) && continue
      o = b.origins[t]
      fill!(buf, 0)
      spread_nodes!(buf, K, b, vals, Ñ, o, r)
      fl(gp, buf, o, L)
    end
  end
  gp
end

# 3D: the buffer slides along dimension D over a segment of a tile column (ducc's shift-and-flush):
# after each tile only the B slices the next tile no longer touches are flushed, the remaining W+1
# slices move to the front. Every grid point is flushed from (E/B)^(D−1) instead of (E/B)^D
# buffers, E = B+W+1 (3D, W = 13: 37.6 → 29.4 ms).
function resample_adjoint_slide!(gp, K::SymKernel{W}, b::Binned{D}, vals, Ñ, L::TileLocks{D}, bufs, fl=flush!) where {W,D}
  ntD = b.ntiles[D]; ncol = prod(Base.front(b.ntiles))
  ntasks = length(bufs); B = L.B
  nseg = clamp(cld(8ntasks, ncol), 1, ntD); seglen = cld(ntD, nseg)   # 8 segments per task for balance
  segs = [(c - 1) * ntD .+ (s:min(s + seglen - 1, ntD)) for c in 1:ncol for s in 1:seglen:ntD]
  E = size(bufs[1], 1)
  next = Threads.Atomic{Int}(1)
  tasks(min(ntasks, length(segs))) do it
    buf = bufs[it]
    S = E^(D - 1)
    while (si = Threads.atomic_add!(next, 1)) <= length(segs)
      tiles = segs[si]
      b.tileptr[last(tiles)+1] > b.tileptr[first(tiles)] || continue
      fill!(buf, 0)
      for t in tiles
        o = b.origins[t]
        spread_nodes!(buf, K, b, vals, Ñ, o, b.tileptr[t]:b.tileptr[t+1]-1)
        if t == last(tiles)
          fl(gp, buf, o, L)
        else
          fl(gp, lastdim_view(buf, 1:B), o, L)
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
  gp
end

# `fl` flushes a buffer into the grid (the ablation switches replace it).
resample_adjoint!(gp, K, b::Binned{D}, vals, Ñ, L, bufs, fl=flush!) where {D} =
  D >= 3 ? resample_adjoint_slide!(gp, K, b, vals, Ñ, L, bufs, fl) :
           resample_adjoint_tiles!(gp, K, b, vals, Ñ, L, bufs, fl)

# Halo of the padded grid, h = halo(W): fill it periodically from the interior before resampling,
# fold it back onto the interior after the adjoint resampling. One dimension after the other (this
# covers edges and corners), threaded over another dimension. `dims` are the dimensions with a halo
# (2:ndims for the batch-first grid of the batch kernels), Ñ[d] the grid size along dimension d of
# gp (points behind the upper halo, see padded_size, are left alone).
function halo_chunks(gp, d, nthreads, dims)
  nd = ndims(gp)
  nd == 1 && return d, [1:size(gp, 1)]
  e = d != nd ? nd : first(dims) != d ? first(dims) : 1
  e, chunks(1:size(gp, e), nthreads)
end
function fill_halo!(gp, h::Tuple, nthreads, dims=1:ndims(gp); Ñs=size(gp) .- (h[1] + h[2]))
  for d in dims
    Ñ = Ñs[d]
    e, cs = halo_chunks(gp, d, nthreads, dims)
    @maybe_threads length(cs) for c in cs
      @inbounds for I in halo_slab(gp, d, e, c, 1:h[1])
        gp[I] = gp[shift(I, d, Ñ)]
      end
      @inbounds for I in halo_slab(gp, d, e, c, Ñ+h[1]+1:Ñ+h[1]+h[2])
        gp[I] = gp[shift(I, d, -Ñ)]
      end
    end
  end
  gp
end
function fold_halo!(gp, h::Tuple, nthreads, dims=1:ndims(gp); Ñs=size(gp) .- (h[1] + h[2]))
  for d in reverse(dims)
    Ñ = Ñs[d]
    e, cs = halo_chunks(gp, d, nthreads, dims)
    @maybe_threads length(cs) for c in cs
      @inbounds for I in halo_slab(gp, d, e, c, 1:h[1])
        gp[shift(I, d, Ñ)] += gp[I]
      end
      @inbounds for I in halo_slab(gp, d, e, c, Ñ+h[1]+1:Ñ+h[1]+h[2])
        gp[shift(I, d, -Ñ)] += gp[I]
      end
    end
  end
  gp
end
# The indices r along dimension d, the chunk c along dimension e (e = d in 1D: one chunk).
halo_slab(gp::AbstractArray{T,N}, d, e, c, r) where {T,N} =
  CartesianIndices(ntuple(i -> i == d ? r : i == e ? c : (1:size(gp, i)), Val(N)))
@inline shift(I::CartesianIndex{N}, d, s) where {N} = I + CartesianIndex(ntuple(i -> i == d ? s : 0, Val(N)))
interior(gp, h::Tuple) = view(gp, map(n -> h[1]+1:n-h[2], size(gp))...)
interior(gp, h::Tuple, Ñ) = view(gp, map(n -> h[1]+1:h[1]+n, Ñ)...)
# the batch-first grid: all of dimension 1
interior_b(gp, h::Tuple) = view(gp, :, map(n -> h[1]+1:n-h[2], Base.tail(size(gp)))...)
