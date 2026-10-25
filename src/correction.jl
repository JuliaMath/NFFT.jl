# Resampling correction (Knopp et al. 2023): divide by the Fourier transform of the window and
# place the kept modes into the corners of the oversampled grid g (unpadded view), and the adjoint
# (crop and divide). `invphi[d]` holds 1/φ̂ for the N[d] kept modes of dimension d.

function resampling_correction_core!(g, f, invphi, N::NTuple{D,Int}, Ñ::NTuple{D,Int}, nthreads) where {D}
  Threads.@threads for r in chunks(1:N[D], nthreads)
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
  Threads.@threads for r in chunks(1:N[D], nthreads)
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
