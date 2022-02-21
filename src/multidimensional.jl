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
    if p.params.blocking
      convolve_blocking!(p, g, fHat)
    else
      convolve_nonblocking!(p, g, fHat)
    end
  else
    convolve_sparse_matrix!(p, g, fHat)
  end
  return
end

function convolve_nonblocking!(p::NFFTPlan{T,D,1}, g::AbstractArray{Complex{T},D}, fHat::StridedVector{U}) where {D,T,U}
  L = Val(2*p.params.m)
  scale = Int(p.params.LUTSize/(p.params.m+2))

  @cthreads for k in 1:p.M
      fHat[k] = _convolve_nonblocking(p, g, L, scale, k)
  end
end

function _precomputeOneNode(p::NFFTPlan{T,D,1}, scale, k, d, L::Val{Z}) where {T,D,Z}
    return _precomputeOneNode(p.windowLinInterp, p.x, p.n, 
                    p.params.m, p.params.σ, scale, k, d, L, p.params.LUTSize) 
end

@generated function _convolve_nonblocking(p::NFFTPlan{T,D,1}, g::AbstractArray{Complex{T},D}, L::Val{Z}, scale, k) where {D,T,Z}
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
    if p.params.blocking
      convolve_adjoint_blocking!(p, fHat, g)
    else
      convolve_adjoint_nonblocking!(p, fHat, g)
    end
  else
    convolve_adjoint_sparse_matrix!(p, fHat, g)
  end
end


function convolve_adjoint_nonblocking!(p::NFFTPlan{T,D,1}, fHat::AbstractVector{U}, g::StridedArray{Complex{T},D}) where {D,T,U}
  fill!(g, zero(T))
  L = Val(2*p.params.m)
  scale = Int(p.params.LUTSize/(p.params.m+2))

  @inbounds @simd for k in 1:p.M
    _convolve_adjoint_nonblocking!(p, fHat, g, L, scale, k)
  end
end

@generated function _convolve_adjoint_nonblocking!(p::NFFTPlan{T,D,1}, fHat::AbstractVector{U}, g::StridedArray{Complex{T},D}, L::Val{Z}, scale, k) where {D,T,U,Z}
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




######## blocked multi-threading #########


function makePolyArrayStatic(p::NFFTPlan)
  if p.params.precompute == POLYNOMIAL
    P = p.windowPolyInterp
    return ntuple(d-> ntuple(g-> P[g,d], size(P,1)), size(P,2))
  else
    return nothing
  end
end

function convolve_blocking!(p::NFFTPlan, g, fHat)
  L = Val(2*p.params.m)
  winPoly = makePolyArrayStatic(p)
  _convolve_blocking!(p, g, fHat, L, winPoly)
end

function _convolve_blocking!(p::NFFTPlan{T,D,1}, g, fHat, L, winPoly) where {D,T}
  scale = Int(p.params.LUTSize/(p.params.m+2))

  @cthreads for l in CartesianIndices(size(p.blocks))
    if !isempty(p.nodesInBlock[l])
      toBlock!(p, g, p.blocks[l], p.blockOffsets[l])
      winTensor = (p.params.precompute == TENSOR) ? p.windowTensor[l] : nothing
      calcOneNode!(p, fHat, p.nodesInBlock[l], p.blocks[l], p.blockOffsets[l], L, 
                   scale, p.idxInBlock[l], winTensor, winPoly)
    end
  end
end

@generated function toBlock!(p::NFFTPlan{T,D,1}, g, block, off) where {T,D}
quote
  if off[1] + 2 < 1
    LA = -(off[1] + 2) + 1 
    offA = size(g,1) - LA 
    offB = -LA
  else
    LA = 0
    offB = off[1] + 1
  end

  if offB + size(block,1) > size(g,1)
    LB = size(g,1) - (offB )  
    offC = -LB
  else
    LB = size(block,1)
  end
  
  @nloops_ $(D-1)  (d->l_{d+1})  (d -> 1:size(block,d+1)) d->begin
    # preexpr
    idx_{d+1} = ( rem(l_{d+1}  + off[d+1] + p.n[d+1], p.n[d+1]) + 1)
  end begin
    # bodyexpr
    @inbounds for l_1 = 1:LA
      idx_1 = l_1 + offA
      (@nref $D block l) = (@nref $D g idx)
    end
    @inbounds for l_1 = (LA+1):LB 
      idx_1 = l_1 + offB
      (@nref $D block l) = (@nref $D g idx)
    end
    @inbounds for l_1 = (LB+1):size(block,1) 
      idx_1 = l_1 + offC
      (@nref $D block l) = (@nref $D g idx)
    end
  end
  return
 end
end

@noinline function calcOneNode!(p::NFFTPlan{T,D,1}, fHat, nodesInBlock, block,
         off, L, scale, idxInBlock, winTensor, winPoly) where {D,T}
  for (kLocal,k) in enumerate(nodesInBlock)
    fHat[k] = _convolve_nonblocking_MT(p, block, off, L, scale, k, kLocal, 
                               idxInBlock, winTensor, winPoly)
  end
  return
end

@generated function _convolve_nonblocking_MT(p::NFFTPlan{T,D,1}, block, off, L::Val{Z}, scale, 
                   k, kLocal, idxInBlock, winTensor, winPoly) where {D,T,Z}
  quote
    @nexprs $(D) d -> ((off_d, tmpWin_d) =  _precomputeOneNodeShifted(p.windowLinInterp, winTensor, winPoly, scale, 
                      kLocal, d, L, idxInBlock) )

    fHat = zero(Complex{T})

    @nexprs 1 d -> prodWin_{$D} = one(T)
    @nloops_ $(D-1)  (d->l_{d+1})  (d -> 1:$Z) d->begin
      # preexpr
      prodWin_{d} = prodWin_{d+1} * tmpWin_{d+1}[l_{d+1}]
      block_idx_{d+1} = off_{d+1} + l_{d+1} 
    end begin
      # bodyexpr
      @inbounds for l_1 = 1:$Z
        block_idx_1 = off_1 + l_1 
        prodWin_0 = prodWin_1 * tmpWin_1[l_1]
        fHat += prodWin_0 * (@nref $D block block_idx)
      end
    end
    return fHat
  end
end


##########################

function convolve_adjoint_blocking!(p::NFFTPlan, fHat, g)
  L = Val(2*p.params.m)
  winPoly = makePolyArrayStatic(p)
  _convolve_adjoint_blocking!(p, fHat, g, L, winPoly)
end

function _convolve_adjoint_blocking!(p::NFFTPlan{T,D,1}, fHat, g, L, winPoly) where {D,T}
  #g .= zero(T)
  ccall(:memset, Ptr{Cvoid}, (Ptr{Cvoid}, Cint, Csize_t), g, 0, sizeof(g))
  scale = Int(p.params.LUTSize/(p.params.m+2))
  
  lk = ReentrantLock()
  @cthreads for l in CartesianIndices(size(p.blocks))
    if !isempty(p.nodesInBlock[l])
      # p.blocks[l] .= zero(T)
      ccall(:memset, Ptr{Cvoid}, (Ptr{Cvoid}, Cint, Csize_t), p.blocks[l], 0, sizeof(p.blocks[l]))

      winTensor = (p.params.precompute == TENSOR) ? p.windowTensor[l] : nothing
      fillBlock!(p, fHat, p.blocks[l], p.nodesInBlock[l], p.blockOffsets[l], L, scale, 
                 p.idxInBlock[l], winTensor, winPoly)
 
      lock(lk) do
        addBlock!(p, g, p.blocks[l], p.blockOffsets[l])
      end
    end
  end
end

@generated function addBlock!(p::NFFTPlan{T,D,1}, g, block, off) where {T,D}
  quote
    # addBlock! needs to wrap indices. The easies is to apply a modulo operation (rem)
    # but doing so for each index is slow. We thus only do this for the outer dimensions
    # For the inner dimension we determine the indices where a wrap occurs. This can happen
    # 0 time, 1 time, or 2 times. The last is an extreme case where the block size matches
    # the dimension of g.
    #
    # The following code divides the range 1:size(block,1) into up to three parts and then
    # applies three dedicated for loops. The wrap indices are LA and LB.

    if off[1] + 2 < 1
      # negative indices: first wrapping
      LA = -(off[1] + 2) + 1 
      offA = size(g,1) - LA 
      offB = -LA
    else
      # no negative indices: skip first sum
      LA = 0
      offB = off[1] + 1
    end

    if offB + size(block,1) > size(g,1)
      # out of bounds indices: second (or first) wrapping
      LB = size(g,1) - (offB )  
      offC = -LB
    else
      # no out of bounds indices
      LB = size(block,1)
    end
    
    @nloops_ $(D-1)  (d->l_{d+1})  (d -> 1:size(block,d+1)) d->begin
      # preexpr
      idx_{d+1} = ( rem(l_{d+1}  + off[d+1] + p.n[d+1], p.n[d+1]) + 1)
    end begin
      # bodyexpr
      @inbounds for l_1 = 1:LA
        idx_1 = l_1 + offA
        (@nref $D g idx) += (@nref $D block l)
      end
      @inbounds for l_1 = (LA+1):LB 
        idx_1 = l_1 + offB
        (@nref $D g idx) += (@nref $D block l)
      end
      @inbounds for l_1 = (LB+1):size(block,1) 
        idx_1 = l_1 + offC
        (@nref $D g idx) += (@nref $D block l)
      end
    end
    return
  end
end

@noinline function fillBlock!(p::NFFTPlan{T,D,1}, fHat, block, nodesInBlock, off, L::Val{Z}, scale, 
                              idxInBlock, winTensor, winPoly) where {T,D,Z}
  if (D >= 3 && Z >= 10) || (D == 2 && Z >= 16) # magic
    for (kLocal,k) in enumerate(nodesInBlock)
      fillOneNodeReal!(p, fHat, block, off, L, scale, k, kLocal, idxInBlock, winTensor, winPoly)
    end
  else
    for (kLocal,k) in enumerate(nodesInBlock)
      fillOneNode!(p, fHat, block, off, L, scale, k, kLocal, idxInBlock, winTensor, winPoly)
    end
  end
  return
end


@generated function fillOneNode!(p::NFFTPlan{T,D,1}, fHat, block, off, L::Val{Z}, scale, 
              k, kLocal, idxInBlock, winTensor, winPoly) where {D,T,Z}
  quote
    fHat_ = fHat[k]

    @nexprs $(D) d -> ((off_d, tmpWin_d) = _precomputeOneNodeShifted(p.windowLinInterp, winTensor, winPoly, scale, kLocal, d, L,
                                             idxInBlock) )

    innerWin = @ntuple $(Z) l -> tmpWin_1[l] * fHat_

    @nexprs 1 d -> prodWin_{$D} = one(T)
    @nloops_ $(D-1)  (d->l_{d+1})  (d -> 1:$Z) d->begin
      # preexpr
      prodWin_{d} = prodWin_{d+1} * tmpWin_{d+1}[l_{d+1}]
      block_idx_{d+1} = off_{d+1} + l_{d+1} 
    end begin
      # bodyexpr
      @inbounds @simd for l_1 = 1:$(Z)
        block_idx_1 = off_1 + l_1 
        (@nref $D block block_idx) += innerWin[l_1] * prodWin_1
      end
    end
    return
  end
end


@generated function fillOneNodeReal!(p::NFFTPlan{T,D,1}, fHat, block, off, L::Val{Z}, scale, k, 
                kLocal, idxInBlock, winTensor, winPoly) where {D,T,Z}
  quote
    fHat_ = fHat[k]

    @nexprs $(D) d -> ((off_d, tmpWin_d) =  _precomputeOneNodeShifted(p.windowLinInterp, winTensor, winPoly, scale, kLocal, 
                                              d, L, idxInBlock) )

    innerWin = Vector{Float64}(undef,$(2*Z)) # Probably cache me somewhere
    for l=1:$Z
      innerWin[2*(l-1)+1] = tmpWin_1[l] * real(fHat_)
      innerWin[2*(l-1)+2] = tmpWin_1[l] * imag(fHat_)
    end
    #innerWin = @ntuple $(2*Z) l -> isodd(l) ? tmpWin_1[(l-1)÷2+1] * real(fHat_) : tmpWin_1[(l-1)÷2+1] * imag(fHat_)

    # Convert to real
    # We use unsafe_wrap here because reinterpret(T, block) is much slower
    # and reinterpret(T, reshape, block) is slightly slower than unsafe_wrap
    # In fact the conversion to real only pays off when using unsafe_wrap, 
    # otherwise sticking with complex is faster
    U = @ntuple $(D) d -> d==1 ? 2*size(block,1) : size(block,d)
    blockReal = unsafe_wrap(Array,reinterpret(Ptr{T},pointer(block)), U)
    off_1 *= 2

    @nexprs 1 d -> prodWin_{$D} = one(T)
    @nloops_ $(D-1)  (d->l_{d+1})  (d -> 1:$Z) d->begin
      # preexpr
      prodWin_{d} = prodWin_{d+1} * tmpWin_{d+1}[l_{d+1}]
      block_idx_{d+1} = off_{d+1} + l_{d+1} 
    end begin
      # bodyexpr
      @inbounds @simd for l_1 = 1:$(2*Z)
        block_idx_1 = off_1 + l_1 
        (@nref $(D) blockReal block_idx) += innerWin[l_1] * prodWin_1
      end
    end
    return
  end
end
