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


function precomputation(x::Union{Matrix{T},Vector{T}}, N::NTuple{D,Int}, n, m = 4, σ = 2.0, window = :kaiser_bessel, K = 2000, precompute::PrecomputeFlags = LUT) where {T,D}

  win, win_hat = getWindow(window)
  M = size(x, 2)

  windowLUT = Vector{Vector{T}}(undef, D)
  windowHatInvLUT = Vector{Vector{T}}(undef, D)
  precomputeWindowHatInvLUT(windowHatInvLUT, win_hat, N, n, m, σ, T)

  if precompute == LUT
      precomputeLUT(win, windowLUT, n, m, σ, K, T)
      B = sparse([],[],T[])
  elseif precompute == FULL
      B = precomputeB(win, x, N, n, m, M, σ, K, T)
  elseif precompute == FULL_LUT
      precomputeLUT(win, windowLUT, n, m, σ, K, T)
      B = precomputeB(windowLUT, x, N, n, m, M, σ, K, T)
  else
      error("precompute = $precompute not supported by NFFT.jl!")
  end
  return (windowLUT, windowHatInvLUT, B)
end

"""
precompute LUT for the multidimensional interpolation window
"""
function precomp_windowHatInvLUT(T::Type, win_hat::Function, N::NTuple{D,Int64}, σ::Real, m::Int64) where D
    # size of oversampled grid
    n = ntuple(d->round(Int,σ*N[d]), D)
    # lookup tables for 1d interpolation kernels
    windowHatInvLUT1d = Vector{Vector{T}}(undef,D)
    for d=1:D
        windowHatInvLUT1d[d] = [1.0/win_hat(k-1-N[d]/2, n[d], m, σ) for k=1:N[d]]
    end
    # lookup table for multi-dimensional kernels
    windowHatInvLUT = zeros(Complex{T},N)
    if D==1
        windowHatInvLUT .= windowHatInvLUT1d[1]
    elseif D==2
        windowHatInvLUT .= reshape( kron(windowHatInvLUT1d[1], windowHatInvLUT1d[2]), N )
    elseif D==3
        windowHatInvLUT2d = kron(windowHatInvLUT1d[1], windowHatInvLUT1d[2])
        windowHatInvLUT .= reshape( kron(windowHatInvLUT2d, windowHatInvLUT1d[3]), N )
    else
        error("CuNFFT does not yet support $(D) dimensions")
    end
    return windowHatInvLUT
end

"""
precompute indices of the apodized image in the oversampled grid
"""
@generated function precomp_apodIdx(N::NTuple{D,Int64}, n::NTuple{D,Int64}) where D
    quote
        # linear indices of the oversampled grid
        linIdx = LinearIndices(n)
        # offsets to central NxN-region of the oversampled grid
        @nexprs $D d -> offset_d = round(Int, n[d] - N[d]/2) - 1
        # apodization indices
        apodIdx = zeros(Int64,N)
        @nloops $D l apodIdx d->(gidx_d = rem(l_d+offset_d, n[d]) + 1) begin
            (@nref $D apodIdx l) = (@nref $D linIdx gidx)
        end
        return vec(apodIdx)
    end
end