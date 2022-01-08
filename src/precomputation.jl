### Precomputation of the B matrix ###

function precomputeB(win, x, N::NTuple{D,Int}, n::NTuple{D,Int}, m, M, σ, K, T) where D
  I = Array{Int64,2}(undef, (2*m+1)^D, M)
  β = (2*m+1)^D
  J = [β*k+1 for k=0:M]
  V = Array{T,2}(undef,(2*m+1)^D, M)
  mProd = ntuple(d-> (d==1) ? 1 : (2*m+1)^(d-1), D)
  nProd = ntuple(d-> (d==1) ? 1 : prod(n[1:(d-1)]), D)
  L = Val(2*m+1)
  scale = T(1.0 / m * (K-1))

  @cthreads for k in 1:M
    _precomputeB(win, x, N, n, m, M, σ, scale, I, J, V, mProd, nProd, L, k)
  end

  S = SparseMatrixCSC(prod(n), M, J, vec(I), vec(V))
  return S
end

@inline @generated function _precomputeB(win, x::AbstractMatrix{T}, N::NTuple{D,Int}, n::NTuple{D,Int}, m, M, 
                     σ, scale, I, J, V, mProd, nProd, L::Val{Z}, k) where {T, D, Z}
  quote

    @nexprs $(D) d -> ((tmpIdx_d, tmpWin_d) = _precomputeOneNode(win, x, n, m, σ, scale, k, d, L) )

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
        I[κ_0,k] = ζ_0
        V[κ_0,k] = prodWin_0
    end
    return
  end
end

### precomputation of the window and the indices required during convolution ###

@generated function _precomputeOneNode(win::Function, x::AbstractMatrix{T}, n::NTuple{D,Int}, m, 
  σ, scale, k, d, L::Val{Z}) where {T,D,Z}
  quote
    xscale = x[d,k] * n[d]
    off = floor(Int, xscale) - m - 1
    tmpIdx = @ntuple $(Z) l -> ( rem(l + off + n[d], n[d]) + 1)
    tmpWin = @ntuple $(Z) l -> (win(abs( (xscale - l - off) )  / n[d], n[d], m, σ) )
    return (tmpIdx, tmpWin)
  end
end

@generated function _precomputeOneNode(windowLUT::Vector, x::AbstractMatrix{T}, n::NTuple{D,Int}, m, 
  σ, scale, k, d, L::Val{Z}) where {T,D,Z}
  quote
    xscale = x[d,k] * n[d]
    off = floor(Int, xscale) - m - 1
    tmpIdx = @ntuple $(Z) l -> ( rem(l + off + n[d], n[d]) + 1)
    tmpWin = @ntuple $(Z) l -> begin
      idx = abs( (xscale - l - off)*scale ) + 1
      idxL = floor(idx)
      idxInt = Int(idxL)
      (windowLUT[d][idxInt] + ( idx-idxL ) * (windowLUT[d][idxInt+1] - windowLUT[d][idxInt]))  
    end
    return (tmpIdx, tmpWin)
  end
end


##################

function precomputeLUT(win, windowLUT, n, m, σ, K, T)
    Z = round(Int, 3 * K / 2)
    for d = 1:length(windowLUT)
        windowLUT[d] = Vector{T}(undef, Z)
        @cthreads for l = 1:Z
            y = ((l - 1) / (K - 1)) * m / n[d]
            windowLUT[d][l] = win(y, n[d], m, σ)
        end
    end
end

function precomputeWindowHatInvLUT(windowHatInvLUT, win_hat, N, n, m, σ, T)
  for d=1:length(windowHatInvLUT)
      windowHatInvLUT[d] = zeros(T, N[d])
      @cthreads for k=1:N[d]
          windowHatInvLUT[d][k] = 1. / win_hat(k-1-N[d]÷2, n[d], m, σ)
      end
  end
end


function precomputation(x::Union{Matrix{T},Vector{T}}, N::NTuple{D,Int}, n, params) where {T,D}

  m = params.m; σ = params.σ; window=params.window 
  LUTSize = params.LUTSize; precompute = params.precompute

  win, win_hat = getWindow(window)
  M = size(x, 2)

  windowLUT = Vector{Vector{T}}(undef, D)

  windowHatInvLUT_ = Vector{Vector{T}}(undef, D)
  precomputeWindowHatInvLUT(windowHatInvLUT_, win_hat, N, n, m, σ, T)

  if params.storeApodizationIdx
    windowHatInvLUT = Vector{Vector{T}}(undef, 1)
    windowHatInvLUT[1], apodizationIdx = precompWindowHatInvLUT(params, N, n, windowHatInvLUT_)  
  else
    windowHatInvLUT = windowHatInvLUT_
    apodizationIdx = Array{Int64,1}(undef, 0)
  end

  if precompute == LUT
      precomputeLUT(win, windowLUT, n, m, σ, LUTSize, T)
      B = sparse([],[],T[])
  elseif precompute == FULL
      B = precomputeB(win, x, N, n, m, M, σ, LUTSize, T)
  elseif precompute == FULL_LUT
      precomputeLUT(win, windowLUT, n, m, σ, LUTSize, T)
      B = precomputeB(windowLUT, x, N, n, m, M, σ, LUTSize, T)
  else
      error("precompute = $precompute not supported by NFFT.jl!")
  end
  return (windowLUT, windowHatInvLUT, apodizationIdx, B)
end

####################################


function precompWindowHatInvLUT(p::NFFTParams{T}, N, n, windowHatInvLUT_) where {T}
  
  windowHatInvLUT = zeros(Complex{T}, N)
  apodIdx = zeros(Int64, N)

  if length(N) == 1
    precompWindowHatInvLUT(p, windowHatInvLUT, apodIdx, N, n, windowHatInvLUT_, 1) 
  else
    @cthreads for o = 1:N[end]
      precompWindowHatInvLUT(p, windowHatInvLUT, apodIdx, N, n, windowHatInvLUT_, o)  
    end
  end
  return vec(windowHatInvLUT), vec(apodIdx)
end

@generated function precompWindowHatInvLUT(p::NFFTParams{T}, windowHatInvLUT::Array{Complex{T},D}, 
           apodIdx::Array{Int,D}, N, n, windowHatInvLUT_, o) where {D,T}
  quote
    linIdx = LinearIndices(n)

    @nexprs 1 d -> gidx_{$D-1} = rem(o+n[$D] - N[$D]÷2 - 1, n[$D]) + 1
    @nexprs 1 d -> l_{$D-1} = o
    @nloops $(D-2) l d->(1:N[d+1]) d-> begin
        gidx_d = rem(l_d+n[d+1] - N[d+1]÷2 - 1, n[d+1]) + 1
      end begin
      N2 = N[1]÷2
      @inbounds @simd for i = 1:N2
        apodIdx[i, CartesianIndex(@ntuple $(D-1) l)] = 
           linIdx[i-N2+n[1], CartesianIndex(@ntuple $(D-1) gidx)]
        v = windowHatInvLUT_[1][i] 
        @nexprs $(D-1) d -> v *= windowHatInvLUT_[d+1][l_d]
        windowHatInvLUT[i, CartesianIndex(@ntuple $(D-1) l)] = v

        apodIdx[i+N2, CartesianIndex(@ntuple $(D-1) l)] = 
           linIdx[i, CartesianIndex(@ntuple $(D-1) gidx)]
        v = windowHatInvLUT_[1][i+N2]
        @nexprs $(D-1) d -> v *= windowHatInvLUT_[d+1][l_d]
        windowHatInvLUT[i+N2, CartesianIndex(@ntuple $(D-1) l)] = v
      end
    end
  end
end
