### Init some initial parameters necessary to create the plan ###

function initParams(k::Matrix{T}, N::NTuple{D,Int}, dims::Union{Integer,UnitRange{Int64}}=1:D; 
                    kargs...) where {D,T}
  # convert dims to a unit range
  dims_ = (typeof(dims) <: Integer) ? (dims:dims) : dims

  params = NFFTParams{T,D}(; kargs...)
  m, σ, reltol = accuracyParams(; kargs...)
  params.m = m
  params.σ = σ
  params.reltol = reltol

  # Taken from NFFT3
  m2K = [1, 3, 7, 9, 14, 17, 20, 23, 24]
  K = m2K[min(m+1,length(m2K))]
  params.LUTSize = 2^(K) * (m) # ensure that LUTSize is dividable by (m)

  if length(dims_) != size(k,1)
      throw(ArgumentError("Nodes x have dimension $(size(k,1)) != $(length(dims_))"))
  end

  doTrafo = ntuple(d->d ∈ dims_, D)

  Ñ = ntuple(d -> doTrafo[d] ?
                      (ceil(Int,params.σ*N[d])÷2)*2 : # ensure that n is an even integer
                        N[d], D)

  params.σ = Ñ[dims_[1]] / N[dims_[1]]

  #params.blockSize = ntuple(d-> Ñ[d] , D) # just one block
  if haskey(kargs, :blockSize)
    params.blockSize = kargs[:blockSize]
  else
    params.blockSize = ntuple(d-> _blockSize(Ñ,d) , D)
  end

  J = size(k, 2)

  # calculate output size
  NOut = Int[]
  Mtaken = false
  for d=1:D
    if !doTrafo[d]
      push!(NOut, N[d])
    elseif !Mtaken
      push!(NOut, J)
      Mtaken = true
    end
  end
  # Sort nodes in lexicographic way
  if params.sortNodes
      k .= sortslices(k, dims=2)
  end
  return params, N, Tuple(NOut), J, Ñ, dims_
end


function _blockSize(Ñ::NTuple{1,Int}, d)
  return min(1024, Ñ[d])
end

function _blockSize(Ñ::NTuple{2,Int}, d)
  return min(64, Ñ[d])
end

function _blockSize(Ñ::NTuple{D,Int}, d) where {D}
  if d == 1
    return min(16, Ñ[d])
  elseif d == 2
    return min(16, Ñ[d])
  elseif d == 3
    return min(16, Ñ[d])
  else
    return 1
  end
end

### Precomputation of the B matrix ###

function precomputeB(win, k, N::NTuple{D,Int}, Ñ::NTuple{D,Int}, m, J, σ, K, T) where D
  I = Array{Int64,2}(undef, (2*m)^D, J)
  β = (2*m)^D
  Y = [β*k+1 for k=0:J]
  V = Array{T,2}(undef,(2*m)^D, J)
  mProd = ntuple(d-> (d==1) ? 1 : (2*m)^(d-1), D)
  nProd = ntuple(d-> (d==1) ? 1 : prod(Ñ[1:(d-1)]), D)
  L = Val(2*m)
  scale = Int(K/(m))

  @cthreads for j in 1:J
    _precomputeB(win, k, N, Ñ, m, J, σ, scale, I, Y, V, mProd, nProd, L, j, K)
  end

  S = SparseMatrixCSC(prod(Ñ), J, Y, vec(I), vec(V))
  return S
end

@inline @generated function _precomputeB(win, k::AbstractMatrix{T}, N::NTuple{D,Int}, Ñ::NTuple{D,Int}, m, J,
                     σ, scale, I, Y, V, mProd, nProd, L::Val{Z}, j, LUTSize) where {T, D, Z}
  quote

    @nexprs $(D) d -> ((tmpIdx_d, tmpWin_d) = precomputeOneNode(win, k, Ñ, m, σ, scale, j, d, L, LUTSize) )

    @nexprs 1 d -> κ_{$D} = 1 # This is a hack, I actually want to write κ_$D = 1
    @nexprs 1 d -> ζ_{$D} = 1
    @nexprs 1 d -> prodWin_{$D} = one(T)
    @nloops $D l d -> 1:$Z d->begin
        # preexpr
        prodWin_{d-1} = prodWin_d * tmpWin_d[l_d]
        κ_{d-1} = κ_d + (l_d-1) * mProd[d]
        ζ_{d-1} = ζ_d + (tmpIdx_d[l_d]-1) * nProd[d]
    end begin
        # bodyexpr
        I[κ_0,j] = ζ_0
        V[κ_0,j] = prodWin_0
    end
    return
  end
end

### precomputation of the window and the indices required during convolution ###

@generated function precomputeOneNode(win::Function, k::AbstractMatrix{T}, Ñ::NTuple{D,Int}, m,
  σ, scale, j, d, L::Val{Z}, LUTSize) where {T,D,Z}
  quote
    kscale = k[d,j] * Ñ[d]
    off = floor(Int, kscale) - m + 1
    tmpIdx = @ntuple $(Z) l -> ( rem(l + off + Ñ[d] - 1, Ñ[d]) + 1)

    tmpWin = @ntuple $(Z) l -> (win( (kscale - (l-1) - off)  / Ñ[d], Ñ[d], m, σ) )
    return (tmpIdx, tmpWin)
  end
end

# precompute = LINEAR

@generated function precomputeOneNode(winLin::Array, winPoly::Nothing, k::AbstractMatrix{T}, Ñ::NTuple{D,Int}, m,
  σ, scale, j, d, L::Val{Z}, LUTSize) where {T,D,Z}
  quote
    kscale = k[d,j] * Ñ[d]
    off = floor(Int, kscale) - m + 1
    tmpIdx = @ntuple $(Z) l -> ( rem(l + off + Ñ[d] - 1, Ñ[d]) + 1)

    idx = ((kscale - off)*LUTSize)/(m)
    tmpWin =  shiftedWindowEntries(winLin, idx, scale, d, L)

    return (tmpIdx, tmpWin)
  end
end

# precompute = POLYNOMIAL

@generated function precomputeOneNode(winLin::Array, winPoly::NTuple{Y, NTuple{X,T}}, k::AbstractMatrix{T}, Ñ::NTuple{D,Int}, m,
  σ, scale, j, d, L::Val{Z}, LUTSize) where {Y,X,T,D,Z}
  quote
    kscale = k[d,j] * Ñ[d]
    off = floor(Int, kscale) - m + 1
    tmpIdx = @ntuple $(Z) l -> ( rem(l + off + Ñ[d] - 1, Ñ[d]) + 1)

    idx = (kscale - off - m + T(0.5))
    tmpWin =  shiftedWindowEntries(winPoly, idx, scale, d, L)

    return (tmpIdx, tmpWin)
  end
end

# precompute = LINEAR

function precomputeOneNodeBlocking(winLin, winTensor::Nothing, winPoly::Nothing,
                                    scale, j, d, L, idxInBlock::Matrix)

  y, idx = idxInBlock[d,j]
  tmpWin =  shiftedWindowEntries(winLin, idx, scale, d, L)

  return (y, tmpWin)
end

@generated function shiftedWindowEntries(winLin::Vector, idx, scale, d, L::Val{Z}) where {Z}
  quote
    idxInt = floor(Int,idx)
    α = ( idx-idxInt )

    tmpWin = @ntuple $(Z) l -> begin
      # Uncommented code: This is the version where we pull in l into the abs.
      # We pulled this out of the iteration.
      # idx = abs((kscale - (l-1)  - off)*LUTSize)/(m)

      # The second +1 is because Julia has 1-based indexing
      # The first +1 is part of the index calculation and needs(!)
      # to be part of the abs. The abs is shifting to the positive interval
      # and this +1 matches the `floor` above, which rounds down. In turn
      # for positive and negative indices a different neighbour is calculated
      idxInt1 = abs( idxInt - (l-1)*scale ) +1
      idxInt2 = abs( idxInt - (l-1)*scale +1) +1

      (winLin[idxInt1] + α * (winLin[idxInt2] - winLin[idxInt1]))
    end
    return tmpWin
  end
end

# precompute = POLYNOMIAL

function precomputeOneNodeBlocking(winLin, winTensor::Nothing, winPoly::NTuple{Z, NTuple{X,T}}, scale,
                                   j, d, L, idxInBlock::Matrix) where {T,Z,X}

  y, k =idxInBlock[d,j]
  tmpWin =  shiftedWindowEntries(winPoly, k, scale, d, L)

  return (y, tmpWin)
end


@generated function shiftedWindowEntries(winPoly::NTuple{Z, NTuple{X,T}}, k::T, scale, d, L::Val{Z}) where {T,Z,X}
  quote
    tmpWin = @ntuple $(Z) l -> begin
      evalpoly(k, winPoly[l])
    end
    return tmpWin
  end
end

#= Static Array version. Not faster
function precomputeOneNodeBlocking(winLin, winTensor::Nothing, winPoly::SMatrix{X,Z,T}, scale,
  j, d, L, idxInBlock::Matrix) where {T,Z,X}

  y, k = idxInBlock[d,j]
  tmpWin =  shiftedWindowEntries(winPoly, k, scale, d, L)

  return (y, tmpWin)
end

@generated function shiftedWindowEntries(winPoly::SMatrix{X,Z,T}, k::T, scale, d, L::Val{Z}) where {T,Z,X}
  quote

    k_1 = one(T)
    @nexprs $(X-1) h->(x_{h+1} = k_h * k)

    xx = @ntuple $(X) h -> begin
      k_{h}
    end

    ks = SVector(xx...)

    tmpWin = transpose(winPoly) * ks

    return tmpWin
  end
end
=#

# precompute = TENSOR

function precomputeOneNodeBlocking(winLin, winTensor::Array, winPoly::Nothing, scale,
                                   j, d, L, idxInBlock::Matrix)
  y, idx = idxInBlock[d,j]
  tmpWin =  shiftedWindowEntriesTensor(winTensor, j, d, L)

  return (y, tmpWin)
end


@generated function shiftedWindowEntriesTensor(winTensor, j, d, L::Val{Z}) where {Z}
  quote
    tmpWin = @ntuple $(Z) l -> begin
      winTensor[l, d, j]
    end
    return tmpWin
  end
end


##################

## function nfft_precompute_lin_psi in NFFT3
"""
Precompute the look up table for the window function φ.

Remarks:
* Only the positive half is computed
* The window is computed for the interval [0, (m)/Ñ]. The reason for the +2 is
  that we do evaluate the window function outside its interval, since x does not
  necessary match the sampling points
* The window has K+1 entries and during the index calculation we multiply with the
  factor K/(m).
* It is very important that K/(m) is an integer since our index calculation exploits
  this fact. We therefore always use `Int(K/(m))`instead of `K÷(m)` since this gives
  an error while the later variant would silently error.
"""
function precomputeLinInterp(win, m, σ, K, T)
  windowLinInterp = Vector{T}(undef, K+2)

  step = (m) / (K)
  @cthreads for l = 1:(K+2)
      y = ( (l-1) * step )
      windowLinInterp[l] = win(y, 1, m, σ)
  end
  return windowLinInterp
end

function precomputePolyInterp(win, m, σ, T)
  deg = 2*m+1 # Certainly depends on Window
  K = 2*m
  NSamples = 2*deg # Sample more densely!!!
  windowPolyInterp = Matrix{T}(undef, deg, K)

  k = range(-0.5, 0.5,length=NSamples)
  V = ones(deg, NSamples)
  for r=2:deg
    V[r,:] .= V[r-1,:] .* k
  end

  for l = 1:K
      y = (-(l-0.5) + m) .+ k
      samples = win.(y, 1, m, σ)
      windowPolyInterp[:,l] .= V' \ samples
  end
  return windowPolyInterp
end

function testPrecomputePoly(win, m, σ, T)
  deg = 2*m + 3
  K = 2*m
  step = 1

  windowPolyInterp = precomputePolyInterp(win, m, σ, T)

  k = -0.394234
  kk = [k^l for l=0:(deg-1)]

  winTrue = zeros(T,K)
  winApprox = zeros(T,K)

  for l = 1:K
    y = (-(l-0.5) * step + m) + k
    @info y
    winTrue[l] = win(y, 1, m, σ)
    winApprox[l] = windowPolyInterp[:,l]' * kk
  end

  return winTrue, winApprox
end

indexOffset(N) = iseven(N) ? (-1-N÷2) : (-1-(N-1)÷2)

function precomputeWindowHatInvLUT(windowHatInvLUT, win_hat, N, Ñ, m, σ, T)

  for d=1:length(windowHatInvLUT)
      κ = n -> win_hat(n + indexOffset(N[d]), Ñ[d], m, σ)
      cheb = ChebyshevInterpolator(κ, 1, N[d], 30)

      windowHatInvLUT[d] = zeros(T, N[d])
      @cthreads for j=1:N[d]
          windowHatInvLUT[d][j] = 1. / cheb(j) #win_hat(k + indexOffset(N[d]), Ñ[d], m, σ)
      end
  end
end

function precomputation(k::Union{Matrix{T},Vector{T}}, N::NTuple{D,Int}, Ñ, params) where {T,D}

  m = params.m; σ = params.σ; window=params.window
  LUTSize = params.LUTSize; precompute = params.precompute

  win, win_hat = getWindow(window) # highly type instable. But what should be do
  J = size(k, 2)

  windowHatInvLUT_ = Vector{Vector{T}}(undef, D)
  precomputeWindowHatInvLUT(windowHatInvLUT_, win_hat, N, Ñ, m, σ, T)

  if params.storeDeconvolutionIdx
    windowHatInvLUT = Vector{Vector{T}}(undef, 1)
    windowHatInvLUT[1], deconvolveIdx = precompWindowHatInvLUT(params, N, Ñ, windowHatInvLUT_)
  else
    windowHatInvLUT = windowHatInvLUT_
    deconvolveIdx = Array{Int64,1}(undef, 0)
  end

  if precompute == LINEAR
    windowLinInterp = precomputeLinInterp(win, m, σ, LUTSize, T)
    windowPolyInterp = Matrix{T}(undef, 0, 0)
    B = sparse([],[],T[])
  elseif precompute == POLYNOMIAL
    windowLinInterp = Vector{T}(undef, 0)
    windowPolyInterp = precomputePolyInterp(win, m, σ, T)
    B = sparse([],[],T[])
  elseif precompute == FULL
    windowLinInterp = Vector{T}(undef, 0)
    windowPolyInterp = Matrix{T}(undef, 0, 0)
    B = precomputeB(win, k, N, Ñ, m, J, σ, LUTSize, T)
    #windowLinInterp = precomputeLinInterp(win, windowLinInterp, Ñ, m, σ, LUTSize, T) # These versions are for debugging
    #B = precomputeB(windowLinInterp, k, N, Ñ, m, J, σ, LUTSize, T)
  elseif precompute == TENSOR
    windowLinInterp = Vector{T}(undef, 0)
    windowPolyInterp = Matrix{T}(undef, 0, 0)
    B = sparse([],[],T[])
  else
    windowLinInterp = Vector{T}(undef, 0)
    windowPolyInterp = Matrix{T}(undef, 0, 0)
    B = sparse([],[],T[])
    error("precompute = $precompute not supported by NFFT.jl!")
  end

  return (windowLinInterp, windowPolyInterp, windowHatInvLUT, deconvolveIdx, B)
end

####################################


# This function is type unstable. why???
function precompWindowHatInvLUT(p::NFFTParams{T}, N, Ñ, windowHatInvLUT_) where {T}

  windowHatInvLUT = zeros(Complex{T}, N)
  deconvIdx = zeros(Int64, N)

  if length(N) == 1
    precompWindowHatInvLUT(p, windowHatInvLUT, deconvIdx, N, Ñ, windowHatInvLUT_, 1)
  else
    @cthreads for o = 1:N[end]
      precompWindowHatInvLUT(p, windowHatInvLUT, deconvIdx, N, Ñ, windowHatInvLUT_, o)
    end
  end
  return vec(windowHatInvLUT), vec(deconvIdx)
end

@generated function precompWindowHatInvLUT(p::NFFTParams{T}, windowHatInvLUT::AbstractArray{Complex{T},D},
           deconvIdx::AbstractArray{Int,D}, N, Ñ, windowHatInvLUT_, o)::Nothing where {D,T}
  quote
    linIdx = LinearIndices(Ñ)

    @nexprs 1 d -> gidx_{$D-1} = rem(o+Ñ[$D] + indexOffset(N[$D]), Ñ[$D]) + 1
    @nexprs 1 d -> l_{$D-1} = o
    @nloops $(D-2) l d->(1:N[d+1]) d-> begin
        gidx_d = rem(l_d+Ñ[d+1] + indexOffset(N[d+1]), Ñ[d+1]) + 1
      end begin
      Na = N[1]÷2
      @inbounds @simd for i = 1:Na
        deconvIdx[i, CartesianIndex(@ntuple $(D-1) l)] =
           linIdx[i-Na+Ñ[1], CartesianIndex(@ntuple $(D-1) gidx)]
        v = windowHatInvLUT_[1][i]
        @nexprs $(D-1) d -> v *= windowHatInvLUT_[d+1][l_d]
        windowHatInvLUT[i, CartesianIndex(@ntuple $(D-1) l)] = v
      end
      Nb = (N[1]+1)÷2
      @inbounds @simd for i = 1:Nb
        deconvIdx[i+Na, CartesianIndex(@ntuple $(D-1) l)] =
           linIdx[i, CartesianIndex(@ntuple $(D-1) gidx)]
        v = windowHatInvLUT_[1][i+Na]
        @nexprs $(D-1) d -> v *= windowHatInvLUT_[d+1][l_d]
        windowHatInvLUT[i+Na, CartesianIndex(@ntuple $(D-1) l)] = v
      end
    end
    return
  end
end

####### block precomputation #########

function precomputeBlocks(k::Matrix{T}, Ñ::NTuple{D,Int}, params, calcBlocks::Bool) where {T,D}

  if calcBlocks
    xShift = copy(k)
    shiftNodes!(xShift)
    blocks, nodesInBlocks, blockOffsets =
        _precomputeBlocks(xShift, Ñ, params.m, params.LUTSize, params.blockSize)

    idxInBlock =  _precomputeIdxInBlock(xShift, Ñ, params.m, params.precompute, params.LUTSize, blockOffsets, nodesInBlocks)
    if params.precompute != TENSOR
      windowTensor = Array{Array{T,3},D}(undef, ntuple(d->0,D))
    else
      #idxInBlock = Array{Matrix{Tuple{Int,Float64}},D}(undef, ntuple(d->0,D))
      windowTensor = _precomputeWindowTensor(xShift, Ñ, params.m, params.σ, nodesInBlocks, params.window)
    end

  else
    blocks = Array{Array{Complex{T},D},D}(undef,ntuple(d->0,D))
    nodesInBlocks = Array{Vector{Int64},D}(undef,ntuple(d->0,D))
    blockOffsets = Array{NTuple{D,Int64},D}(undef,ntuple(d->0,D))
    idxInBlock = Array{Matrix{Tuple{Int,T}},D}(undef, ntuple(d->0,D))
    windowTensor = Array{Array{T,3},D}(undef, ntuple(d->0,D))
  end

  return (blocks, nodesInBlocks, blockOffsets, idxInBlock, windowTensor)
end


function _precomputeBlocks(k::Matrix{T}, Ñ::NTuple{D,Int}, m, LUTSize, blockSize) where {T,D}

  padding = ntuple(d->m, D)
  numBlocks =  ntuple(d-> ceil(Int, Ñ[d]/blockSize[d]), D)
  blockSizePadded = ntuple(d-> blockSize[d] + 2*padding[d], D)
  nodesInBlock = [ Int[] for l in CartesianIndices(numBlocks) ]
  numNodesInBlock = zeros(Int, numBlocks)
  for j=1:size(k,2) # @cthreads
    idx = ntuple(d->unsafe_trunc(Int, k[d,j]*Ñ[d])÷blockSize[d]+1, D)
    numNodesInBlock[idx...] += 1
  end
  @cthreads  for l in CartesianIndices(numBlocks)
    sizehint!(nodesInBlock[l], numNodesInBlock[l])
  end
  for j=1:size(k,2) # @cthreads
    idx = ntuple(d->unsafe_trunc(Int, k[d,j]*Ñ[d])÷blockSize[d]+1, D)
    push!(nodesInBlock[idx...], j)
  end

  blocks = Array{Array{Complex{T},D},D}(undef, numBlocks)
  blockOffsets = Array{NTuple{D,Int64},D}(undef, numBlocks)

  @cthreads for l in CartesianIndices(numBlocks)
    if !isempty(nodesInBlock[l])
      # precompute blocks
      blocks[l] = Array{Complex{T},D}(undef, blockSizePadded)

      # precompute blockOffsets
      blockOffsets[l] = ntuple(d-> (l[d]-1)*blockSize[d]-padding[d]-1, D)
    end
  end

  return blocks, nodesInBlock, blockOffsets
end



function _precomputeIdxInBlock(k::Matrix{T}, Ñ::NTuple{D,Int}, m, precompute, LUTSize, blockOffsets, nodesInBlock) where {T,D}

  numBlocks = size(nodesInBlock)

  idxInBlock = Array{Matrix{Tuple{Int,T}},D}(undef, numBlocks)

  @cthreads for l in CartesianIndices(numBlocks)
    if !isempty(nodesInBlock[l])

      # precompute idxInBlock
      idxInBlock[l] = Matrix{Tuple{Int,T}}(undef, D, length(nodesInBlock[l]))
      @inbounds for (i,j) in enumerate(nodesInBlock[l])
        @inbounds for d=1:D
          xtmp = k[d,j] # this is expensive because of cache misses
          kscale = xtmp * Ñ[d]
          off = unsafe_trunc(Int, kscale) - m + 1
          y = off - blockOffsets[l][d] - 1

          if precompute == LINEAR
            idx = (kscale - off)*(LUTSize÷(m))
          else
            idx = (kscale - off - m + 1 -0.5 )
          end
          idxInBlock[l][d,i] = (y,idx)
        end
      end
    end
  end

  return idxInBlock
end

function _precomputeWindowTensor(k::Matrix{T}, Ñ::NTuple{D,Int}, m, σ, nodesInBlock, window::Symbol) where {T,D}
  win, win_hat = getWindow(window) # highly type instable. But what should be do
  P = precomputePolyInterp(win, m, σ, T)
  winPoly = ntuple(d-> ntuple(g-> P[g,d], size(P,1)), size(P,2))

  return _precomputeWindowTensor(k, Ñ, m, σ, nodesInBlock, winPoly)
end


function _precomputeWindowTensor(k::Matrix{T}, Ñ::NTuple{D,Int}, m, σ, nodesInBlock, winPoly::NTuple{Z, NTuple{X,T}}) where {T,D,Z,X}

  numBlocks = size(nodesInBlock)
  windowTensor = Array{Array{T,3},D}(undef, numBlocks)

  @cthreads for l in CartesianIndices(numBlocks)
    if !isempty(nodesInBlock[l])

      # precompute idxInBlock
      windowTensor[l] = Array{T,3}(undef, 2*m, D, length(nodesInBlock[l]))
      @inbounds for (i,j) in enumerate(nodesInBlock[l])
        @inbounds for d=1:D
          xtmp = k[d,j]  # this is expensive because of cache misses
          kscale = xtmp * Ñ[d]
          off = unsafe_trunc(Int, kscale) - m + 1
          k_ = (kscale - off - m + 1 -0.5 )

          @inbounds @simd for j=1:Z
            windowTensor[l][j,d,i] = evalpoly(k_, winPoly[j])
          end
        end
      end
    end
  end

  return windowTensor
end
