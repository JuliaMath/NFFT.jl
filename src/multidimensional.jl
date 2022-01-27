########## apodization ##########

function AbstractNFFTs.apodization!(p::NFFTPlan{T,D,1}, f::AbstractArray{U,D}, g::StridedArray{Complex{T},D}) where {D,T,U}
  if !p.params.storeApodizationIdx
    apodization_alloc_free!(p, f, g)
  else
    p.tmpVecHat[:] .= vec(f) .* p.windowHatInvLUT[1]
    g[p.apodizationIdx] = p.tmpVecHat
  end
  return
end

function apodization_alloc_free!(p::NFFTPlan{T,D,1}, f::AbstractArray{U,D}, g::StridedArray{Complex{T},D}) where {D,T,U}
  if D == 1
    _apodization_alloc_free!(p, f, g, 1) 
  else
    @cthreads for o = 1:p.N[end]
        _apodization_alloc_free!(p, f, g, o)  
    end
  end
end

@generated function _apodization_alloc_free!(p::NFFTPlan{T,D,1}, f::AbstractArray{U,D}, g::StridedArray{Complex{T},D}, o) where {D,T,U}
  quote
    @nexprs 1 d -> gidx_{$D-1} = rem(o+p.n[$D] - p.N[$D]÷2 - 1, p.n[$D]) + 1
    @nexprs 1 d -> l_{$D-1} = o
    @nloops $(D-2) l d->(1:size(f,d+1)) d-> begin
        gidx_d = rem(l_d+p.n[d+1] - p.N[d+1]÷2 - 1, p.n[d+1]) + 1
      end begin
      N2 = p.N[1]÷2
      @inbounds @simd for i = 1:N2
        v = f[i, CartesianIndex(@ntuple $(D-1) l)]
        v *= p.windowHatInvLUT[1][i] 
        @nexprs $(D-1) d -> v *= p.windowHatInvLUT[d+1][l_d]
        g[i-N2+p.n[1], CartesianIndex(@ntuple $(D-1) gidx)] = v

        v = f[i+N2, CartesianIndex(@ntuple $(D-1) l)] 
        v *= p.windowHatInvLUT[1][i+N2]
        @nexprs $(D-1) d -> v *= p.windowHatInvLUT[d+1][l_d]
        g[i, CartesianIndex(@ntuple $(D-1) gidx)] = v
      end
    end
  end
end

########## apodization adjoint ##########

function AbstractNFFTs.apodization_adjoint!(p::NFFTPlan{T,D,1}, g::AbstractArray{Complex{T},D}, f::StridedArray{U,D}) where {D,T,U}
  if !p.params.storeApodizationIdx
    apodization_adjoint_alloc_free!(p, g, f)
  else
    p.tmpVecHat[:] = g[p.apodizationIdx]
    f[:] .= vec(p.tmpVecHat) .* p.windowHatInvLUT[1]
  end
  return
end

function apodization_adjoint_alloc_free!(p::NFFTPlan{T,D,1}, g::AbstractArray{Complex{T},D}, f::StridedArray{U,D}) where {D,T,U}
  if D == 1
    _apodization_adjoint_alloc_free!(p, g, f, 1)  
  else
    @cthreads for o = 1:p.N[end]
      _apodization_adjoint_alloc_free!(p, g, f, o)  
    end
  end
end

@generated function _apodization_adjoint_alloc_free!(p::NFFTPlan{T,D,1}, g::AbstractArray{Complex{T},D}, f::StridedArray{U,D}, o) where {D,T,U}
  quote
    @nexprs 1 d -> gidx_{$D-1} = rem(o+p.n[$D] - p.N[$D]÷2 - 1, p.n[$D]) + 1
    @nexprs 1 d -> l_{$D-1} = o
    @nloops $(D-2) l d->(1:size(f,d+1)) d-> begin
        gidx_d = rem(l_d+p.n[d+1] - p.N[d+1]÷2 - 1, p.n[d+1]) + 1
      end begin
      N2 = p.N[1]÷2
      @inbounds @simd for i = 1:N2
        v = g[i-N2+p.n[1], CartesianIndex(@ntuple $(D-1) gidx)]
        v *= p.windowHatInvLUT[1][i] 
        @nexprs $(D-1) d -> v *= p.windowHatInvLUT[d+1][l_d]
        f[i, CartesianIndex(@ntuple $(D-1) l)] = v

        v = g[i, CartesianIndex(@ntuple $(D-1) gidx)] 
        v *= p.windowHatInvLUT[1][i+N2]
        @nexprs $(D-1) d -> v *= p.windowHatInvLUT[d+1][l_d]
        f[i+N2, CartesianIndex(@ntuple $(D-1) l)] = v
      end
    end
  end
end

########## convolve ##########

function AbstractNFFTs.convolve!(p::NFFTPlan{T,D,1}, g::AbstractArray{Complex{T},D}, fHat::StridedVector{U}) where {D,T,U}
  if isempty(p.B)
    convolve_LUT!(p, g, fHat)
  else
    convolve_sparse_matrix!(p, g, fHat)
  end
end

function convolve_LUT!(p::NFFTPlan{T,D,1}, g::AbstractArray{Complex{T},D}, fHat::StridedVector{U}) where {D,T,U}
  L = Val(2*p.params.m+1)
  scale = T(1.0 / p.params.m * (p.params.LUTSize-1))

  @cthreads for k in 1:p.M
      fHat[k] = _convolve_LUT(p, g, L, scale, k)
  end
end

function _precomputeOneNode(p::NFFTPlan{T,D,1}, scale, k, d, L::Val{Z}) where {T,D,Z}
    return _precomputeOneNode(p.windowLUT, p.x, p.n, p.params.m, p.params.σ, scale, k, d, L) 
end

@generated function _convolve_LUT(p::NFFTPlan{T,D,1}, g::AbstractArray{Complex{T},D}, L::Val{Z}, scale, k) where {D,T,Z}
  quote
    @nexprs $(D) d -> ((tmpIdx_d, tmpWin_d) = _precomputeOneNode(p, scale, k, d, L) )
  
    fHat = zero(Complex{T})

    @nexprs 1 d -> prodWin_{$D} = one(T)
    @nloops_ $D l d -> 1:$Z d->begin
      # preexpr
      prodWin_{d-1} = prodWin_d * tmpWin_d[l_d]
      gidx_d = tmpIdx_d[l_d] 
    end begin
      # bodyexpr
      fHat += prodWin_0 * @nref $D g gidx
    end
    return fHat
  end
end

function convolve_sparse_matrix!(p::NFFTPlan{T,D,1}, g::AbstractArray{Complex{T},D}, fHat::StridedVector{U}) where {D,T,U}
  threaded_mul!(fHat, transpose(p.B), vec(g))
  #mul!(fHat, transpose(p.B), vec(g))
end

########## convolve adjoint ##########

function AbstractNFFTs.convolve_adjoint!(p::NFFTPlan{T,D,1}, fHat::AbstractVector{U}, g::StridedArray{Complex{T},D}) where {D,T,U}
  if isempty(p.B)
    if NFFT._use_threads[]
      convolve_adjoint_LUT_MT!(p, fHat, g)
    else
      convolve_adjoint_LUT!(p, fHat, g)
    end
  else
    convolve_adjoint_sparse_matrix!(p, fHat, g)
  end
end


function convolve_adjoint_LUT!(p::NFFTPlan{T,D,1}, fHat::AbstractVector{U}, g::StridedArray{Complex{T},D}) where {D,T,U}
  fill!(g, zero(T))
  L = Val(2*p.params.m+1)
  scale = T(1.0 / p.params.m * (p.params.LUTSize-1))

  @inbounds @simd for k in 1:p.M
    _convolve_adjoint_LUT!(p, fHat, g, L, scale, k)
  end
end

@generated function _convolve_adjoint_LUT!(p::NFFTPlan{T,D,1}, fHat::AbstractVector{U}, g::StridedArray{Complex{T},D}, L::Val{Z}, scale, k) where {D,T,U,Z}
  quote
    @nexprs $(D) d -> ((tmpIdx_d, tmpWin_d) = _precomputeOneNode(p, scale, k, d, L) )

    @nexprs 1 d -> prodWin_{$D} = one(T)
    @nloops_ $D l d -> 1:$Z d->begin
      # preexpr
      prodWin_{d-1} = prodWin_d * tmpWin_d[l_d]
      gidx_d = tmpIdx_d[l_d] 
    end begin
      # bodyexpr
      (@nref $D g gidx) += prodWin_0 * fHat[k] 
    end
  end
end

function convolve_adjoint_sparse_matrix!(p::NFFTPlan{T,D,1},
                        fHat::AbstractVector{U}, g::StridedArray{Complex{T},D}) where {D,T,U}
  #threaded_mul!(vec(g), p.B, fHat)
  mul!(vec(g), p.B, fHat)
end






function convolve_adjoint_LUT_MT!(p::NFFTPlan{T,D,1}, fHat::AbstractVector{U}, g::StridedArray{Complex{T},D}) where {D,T,U}
  fill!(g, zero(T))
  L = Val(2*p.params.m+1)
  scale = T(1.0 / p.params.m * (p.params.LUTSize-1))

  
  x = p.x
  n = p.n
  
  padding = ntuple(d->p.params.m, D)
  blockSize = ntuple(d-> (d==1) ? 16 : 4 , D)
  blockSizePadded = ntuple(d-> blockSize[d] + 2*padding[d] , D)
  
  numBlocks =  ntuple(d-> ceil(Int, n[d]/blockSize[d]) , D)
  totalBlocks = prod(numBlocks)

  nodesInBlock = [ Int[] for l in CartesianIndices(numBlocks) ]
  
  patches = [ zeros(ComplexF64, blockSizePadded) for l in CartesianIndices(numBlocks) ]
  off = [ ntuple(d-> (l[d]-1)*blockSize[d]+1-padding , D) for l in CartesianIndices(numBlocks) ]
  
 # xscale = x[d,k] * n[d]
 # off = floor(Int, xscale) - m - 1
 # tmpIdx = @ntuple $(Z) l -> ( rem(l + off + n[d], n[d]) + 1)

  for k=1:size(x,2)
    idx = ntuple(d->floor(Int, rem(floor(Int, x[d,k]*n[d])+n[d],n[d])÷blockSize[d])+1, D)
    push!(nodesInBlock[idx...], k)
  end
  
  for l in CartesianIndices(numBlocks)
    if !isempty(nodesInBlock[l])
      fillPatch!(p, fHat, patches[l], nodesInBlock[l], L, scale)
      addPatch!(p, g, patches[l], off[l])
    end
  end
end

function addPatch!(p{T,D,1}, g, patch, off) where {T,D}
  for l in CartesianIndices(size(patch))
    z = ntuple(d->( ( rem(l[d] + off[d] + n[d], n[d]) + 1)  ), D) 
    g[z] = patch[l]
  end
end

function fillPatch!(p::NFFTPlan{T,D,1}, fHat::AbstractVector, patch, nodesInBlock, off, L, scale)  where {D,T,U}
  for k in nodesInBlock
    fillOneNode!(p, fHat, patch, off, L, scale, k)
  end
end



@generated function fillOneNode!(p::NFFTPlan{T,D,1}, fHat::AbstractVector{U}, patch::StridedArray{Complex{T},D},
                          off, L::Val{Z}, scale, k) where {D,T,U,Z}
  quote
    @nexprs $(D) d -> ((tmpIdx_d, tmpWin_d) = _precomputeOneNodeShifted(p, scale, k, d, L, off) )

    @nexprs 1 d -> prodWin_{$D} = one(T)
    @nloops_ $D l d -> 1:$Z d->begin
      # preexpr
      prodWin_{d-1} = prodWin_d * tmpWin_d[l_d]
      patch_idx_d = tmpIdx_d[l_d] 
    end begin
      # bodyexpr
      (@nref $D patch patch_idx) += prodWin_0 * fHat[k] 
    end
  end
end


@generated function _precomputeOneNodeShifted(windowLUT::Vector, x::AbstractMatrix{T}, n::NTuple{D,Int}, m, 
  σ, scale, k, d, L::Val{Z}, off_) where {T,D,Z}
  quote
    xscale = rem(x[d,k]+1.0, 1.0) * n[d]
    off = floor(Int, xscale) # - m - 1
    tmpIdx = @ntuple $(Z) l -> ( l + off + 1)
    tmpWin = @ntuple $(Z) l -> begin
      idx = abs( (xscale - l - off)*scale ) + 1
      idxL = floor(idx)
      idxInt = Int(idxL)
      (windowLUT[d][idxInt] + ( idx-idxL ) * (windowLUT[d][idxInt+1] - windowLUT[d][idxInt]))  
    end
    return (tmpIdx, tmpWin)
  end
end
