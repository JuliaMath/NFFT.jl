
function AbstractNFFTs.deconvolve!(p::NFFTPlan{T,D,1}, f::AbstractArray{U,D}, g::StridedArray{Complex{T},D}) where {D,T,U}
  if !p.params.storeDeconvolutionIdx
    deconvolve_alloc_free!(p, f, g)
  else
    p.tmpVecHat[:] .= vec(f) .* p.windowHatInvLUT[1]
    g[p.deconvolveIdx] = p.tmpVecHat
  end
  return
end

function deconvolve_alloc_free!(p::NFFTPlan{T,D,1}, f::AbstractArray{U,D}, g::StridedArray{Complex{T},D}) where {D,T,U}
  if D == 1
    _deconvolve_alloc_free!(p, f, g, 1) 
  else
    @cthreads for o = 1:p.N[end]
        _deconvolve_alloc_free!(p, f, g, o)  
    end
  end
end

@generated function _deconvolve_alloc_free!(p::NFFTPlan{T,D,1}, f::AbstractArray{U,D}, g::StridedArray{Complex{T},D}, o) where {D,T,U}
  quote
    @nexprs 1 d -> gidx_{$D-1} = rem(o+p.Ñ[$D] + indexOffset(p.N[$D]), p.Ñ[$D]) + 1
    @nexprs 1 d -> l_{$D-1} = o
    @nloops $(D-2) l d->(1:size(f,d+1)) d-> begin
        gidx_d = rem(l_d+p.Ñ[d+1] + indexOffset(p.N[d+1]), p.Ñ[d+1]) + 1
      end begin
      Na = p.N[1]÷2
      @inbounds @simd for i = 1:Na
        v = f[i, CartesianIndex(@ntuple $(D-1) l)]
        v *= p.windowHatInvLUT[1][i] 
        @nexprs $(D-1) d -> v *= p.windowHatInvLUT[d+1][l_d]
        g[i-Na+p.Ñ[1], CartesianIndex(@ntuple $(D-1) gidx)] = v
      end
      Nb = (p.N[1]+1)÷2
      @inbounds @simd for i = 1:Nb
        v = f[i+Na, CartesianIndex(@ntuple $(D-1) l)] 
        v *= p.windowHatInvLUT[1][i+Na]
        @nexprs $(D-1) d -> v *= p.windowHatInvLUT[d+1][l_d]
        g[i, CartesianIndex(@ntuple $(D-1) gidx)] = v
      end
    end
  end
end

########## deconvolve adjoint ##########

function AbstractNFFTs.deconvolve_transpose!(p::NFFTPlan{T,D,1}, g::AbstractArray{Complex{T},D}, f::StridedArray{U,D}) where {D,T,U}
  if !p.params.storeDeconvolutionIdx
    deconvolve_transpose_alloc_free!(p, g, f)
  else
    p.tmpVecHat[:] = g[p.deconvolveIdx]
    f[:] .= vec(p.tmpVecHat) .* p.windowHatInvLUT[1]
  end
  return
end

function deconvolve_transpose_alloc_free!(p::NFFTPlan{T,D,1}, g::AbstractArray{Complex{T},D}, f::StridedArray{U,D}) where {D,T,U}
  if D == 1
    _deconvolve_transpose_alloc_free!(p, g, f, 1)  
  else
    @cthreads for o = 1:p.N[end]
      _deconvolve_transpose_alloc_free!(p, g, f, o)  
    end
  end
end

@generated function _deconvolve_transpose_alloc_free!(p::NFFTPlan{T,D,1}, g::AbstractArray{Complex{T},D}, f::StridedArray{U,D}, o) where {D,T,U}
  quote
    @nexprs 1 d -> gidx_{$D-1} = rem(o+p.Ñ[$D] + indexOffset(p.N[$D]), p.Ñ[$D]) + 1
    @nexprs 1 d -> l_{$D-1} = o
    @nloops $(D-2) l d->(1:size(f,d+1)) d-> begin
        gidx_d = rem(l_d+p.Ñ[d+1] + indexOffset(p.N[d+1]), p.Ñ[d+1]) + 1
      end begin
      Na = p.N[1]÷2
      @inbounds @simd for i = 1:Na
        v = g[i-Na+p.Ñ[1], CartesianIndex(@ntuple $(D-1) gidx)]
        v *= p.windowHatInvLUT[1][i] 
        @nexprs $(D-1) d -> v *= p.windowHatInvLUT[d+1][l_d]
        f[i, CartesianIndex(@ntuple $(D-1) l)] = v
      end
      Nb = (p.N[1]+1)÷2
      @inbounds @simd for i = 1:Nb
        v = g[i, CartesianIndex(@ntuple $(D-1) gidx)] 
        v *= p.windowHatInvLUT[1][i+Na]
        @nexprs $(D-1) d -> v *= p.windowHatInvLUT[d+1][l_d]
        f[i+Na, CartesianIndex(@ntuple $(D-1) l)] = v
      end
    end
  end
end
