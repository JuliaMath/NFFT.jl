# Resampling correction (Knopp et al. 2023): divide by the Fourier transform of the window and
# place the kept modes into the corners of the oversampled grid g (unpadded view), and the adjoint
# (crop and divide). `invphi[d]` holds 1/φ̂ for the N[d] kept modes of dimension d.

# The input of the forward FFT is zero in the interior outside the corners of the kept modes, which
# the correction overwrites; the halo is overwritten by fill_halo!. So the trafo zeroes only that
# region (as FINUFFT does), as boxes of the padded grid (size sz, halo h): along dimension d the
# modes not kept, the dimensions after d kept, those before d whole, including the halo, so that a
# box is contiguous in memory for each index of the dimensions after d. Returned as linear index
# ranges, adjacent ones merged, in about 4 nthreads groups of similar length for zero_runs!.
function unkept_runs(N::NTuple{D,Int}, Ñ::NTuple{D,Int}, h, sz, nthreads) where {D}
  LI = LinearIndices(sz)
  runs = UnitRange{Int}[]
  for d in D:-1:1
    mid = h[1] .+ (N[d]-N[d]÷2+1:Ñ[d]-N[d]÷2)
    isempty(mid) && continue
    lo = ntuple(e -> e < d ? 1 : first(mid), d); hi = ntuple(e -> e < d ? sz[e] : last(mid), d)
    for kept in Iterators.product((filter(!isempty, collect(keptranges(N[e], Ñ[e]))) for e in d+1:D)...),
        J in CartesianIndices(map(r -> h[1] .+ r, kept))
      push!(runs, LI[lo..., Tuple(J)...]:LI[hi..., Tuple(J)...])
    end
  end
  sort!(runs; by=first)
  merged = UnitRange{Int}[]
  for r in runs
    !isempty(merged) && last(merged[end]) + 1 == first(r) ? (merged[end] = first(merged[end]):last(r)) : push!(merged, r)
  end
  runs = merged
  # groups of about total / (4 nthreads) points (long runs split)
  total = sum(length, runs; init=0); target = max(cld(total, 4nthreads), 1)
  groups = [UnitRange{Int}[]]; n = 0
  for r in runs, c in chunks(r, cld(length(r), target))
    n >= target && (push!(groups, UnitRange{Int}[]); n = 0)
    push!(groups[end], c); n += length(c)
  end
  groups
end

# memset on IEEE floats (fill_range!): on a grid that is not in the cache (after the resampling),
# libc's memset zeroes long runs without reading the cache lines (M1: `dc zva`). The short runs
# inside the kept columns lose part of that: at 1 thread in 2D and 3D these runs take 1.2–1.6×
# one memset of the whole padded grid, at 8 threads 0.6–1.0×, in 1D 0.2–0.9×.
function zero_runs!(gp, groups, nthreads)
  @maybe_threads nthreads for g in groups
    for r in g
      fill_range!(gp, zero(eltype(gp)), r)
    end
  end
  gp
end

function resampling_correction_core!(g, f, invphi, N::NTuple{D,Int}, Ñ::NTuple{D,Int}, nthreads) where {D}
  @maybe_threads nthreads for r in chunks(1:N[D], nthreads)
    @inbounds for il in r
      jl = gridindex(il, N[D], Ñ[D]); cl = invphi[D][il]
      for I in CartesianIndices(Base.front(N))
        c = cl
        for d in 1:D-1
          c *= invphi[d][I[d]]
        end
        g[CartesianIndex(ntuple(d -> gridindex(I[d], N[d], Ñ[d]), D - 1)..., jl)] = c * f[I, il]
      end
    end
  end
  g
end

function resampling_correction_adjoint_core!(f, g, invphi, N::NTuple{D,Int}, Ñ::NTuple{D,Int}, nthreads) where {D}
  @maybe_threads nthreads for r in chunks(1:N[D], nthreads)
    @inbounds for il in r
      jl = gridindex(il, N[D], Ñ[D]); cl = invphi[D][il]
      for I in CartesianIndices(Base.front(N))
        c = cl
        for d in 1:D-1
          c *= invphi[d][I[d]]
        end
        f[I, il] = c * g[CartesianIndex(ntuple(d -> gridindex(I[d], N[d], Ñ[d]), D - 1)..., jl)]
      end
    end
  end
  f
end
