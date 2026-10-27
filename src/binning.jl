# Nodes binned into tiles of edge B by a parallel counting sort. Tiles are numbered with the last
# dimension running fastest, so a column of tiles along dimension D is a contiguous range of tiles
# and of sorted nodes (the sliding buffer of the adjoint resampling walks such columns).
struct Binned{D,T}
  xs::Vector{NTuple{D,T}}          # node coordinates in sorted order, in the precision given
  perm::Vector{Int}                # xs[q] is node perm[q] of the input
  tileptr::Vector{Int}             # nodes of tile t are tileptr[t]:tileptr[t+1]-1
  ntiles::NTuple{D,Int}
  origins::Vector{NTuple{D,Int}}   # tile t covers grid positions origins[t] .+ (0:B-1)
  # tiles with dimension 1 fastest, the order for the resampling: neighbouring tasks copy
  # neighbouring blocks of memory (3D, W = 10: 8.5 ms instead of 10.2 ms in column order)
  order1::Vector{Int}
end

nodetype(::Binned{D,T}) where {D,T} = T

function Binned(k::AbstractMatrix{T}, Ñ::NTuple{D,Int}, B::Int, nthreads) where {D,T}
  J = size(k, 2)
  ntiles = cld.(Ñ, B)
  nt = prod(ntiles)
  L = LinearIndices(reverse(ntiles))
  # tile of the grid position (gridpos, as in support); tiles at the periodic seam do not interact
  # because the halo keeps their writes apart in memory
  @inline tile(j) = L[CartesianIndex(reverse(ntuple(d -> min(floor(Int, gridpos(k[d, j], Ñ[d]) / B), ntiles[d] - 1) + 1, D)))]
  cks = chunks(1:J, nthreads)
  t = Vector{Int32}(undef, J)
  counts = zeros(Int, nt, length(cks))
  @maybe_threads nthreads for c in eachindex(cks)
    @inbounds for j in cks[c]
      ti = tile(j); t[j] = ti; counts[ti, c] += 1
    end
  end
  tileptr = Vector{Int}(undef, nt + 1)
  pos = similar(counts)
  acc = 1
  @inbounds for ti in 1:nt
    tileptr[ti] = acc
    for c in eachindex(cks)
      pos[ti, c] = acc; acc += counts[ti, c]
    end
  end
  tileptr[nt+1] = acc
  perm = Vector{Int}(undef, J)
  xs = Vector{NTuple{D,T}}(undef, J)
  @maybe_threads nthreads for c in eachindex(cks)
    @inbounds for j in cks[c]
      ti = t[j]; q = pos[ti, c]; pos[ti, c] = q + 1
      perm[q] = j; xs[q] = ntuple(d -> k[d, j], D)
    end
  end
  origins = Vector{NTuple{D,Int}}(undef, nt)
  order1 = Vector{Int}(undef, nt)
  for (i, I) in enumerate(CartesianIndices(ntiles))
    ti = L[CartesianIndex(reverse(Tuple(I)))]
    origins[ti] = (Tuple(I) .- 1) .* B
    order1[i] = ti
  end
  Binned{D,T}(xs, perm, tileptr, ntiles, origins, order1)
end

# Tile edge (grid points). The resampling wants small tiles (buffers in L1/L2), the adjoint
# resampling large ones (zeroing and flushing (B+W+1)^D buffer points per tile); measured best
# compromise on an M1 Max with 8 threads.
default_tile(D, W) = D == 1 ? 1024 : D == 2 ? 64 : W <= 6 ? 16 : W <= 10 ? 12 : 10

# One lock per tile of edge B of the padded grid, for flushing the private buffers of the adjoint resampling.
struct TileLocks{D}
  locks::Array{Threads.SpinLock,D}
  B::Int
end
TileLocks(sz::NTuple{D,Int}, B) where {D} = TileLocks{D}([Threads.SpinLock() for _ in CartesianIndices(cld.(sz, B))], B)
