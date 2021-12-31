function precomputeB(win, x, N::NTuple{D,Int}, n::NTuple{D,Int}, m, M, sigma, K, T) where D
  I = Array{Int64,2}(undef, (2*m+1)^D, M)
  β = (2*m+1)^D
  J = [β*k+1 for k=0:M]
  V = Array{T,2}(undef,(2*m+1)^D, M)

  @cthreads for k in 1:M
    _precomputeB(win, x, N, n, m, M, sigma, K, T, I, J, V, k)
  end

  S = SparseMatrixCSC(prod(n), M, J, vec(I), vec(V))
  return S
end

## precompute = FULL

@generated function _precomputeB(win::Function, x, N::NTuple{D,Int}, n::NTuple{D,Int}, m, M, sigma, K, T, I,J,V,k) where D
  quote
    @nexprs $D d->(xscale_d = x[d,k] * n[d])
    @nexprs $D d->(c_d = floor(Int, xscale_d))

    N_1 = 1
    @nexprs $D d->(N_{d+1} = N_d * (2*m+1))
    n_1 = 1
    @nexprs $D d->(n_{d+1} = n_d * n[d])

    @nexprs 1 d -> κ_{$D} = 1 # This is a hack, I actually want to write κ_$D = 1
    @nexprs 1 d -> ζ_{$D} = 1
    @nexprs 1 d -> tmpWin_{$D} = one(T)
    @nloops $D l d -> 1:(2*m+1) d->begin
        # preexpr
        off = c_d - m - 1
        gidx_d = rem(l_d  + c_d - m - 1 + n[d], n[d]) + 1
        idx = abs( (xscale_d - l_d - off) )
        tmpWin_{d-1} = tmpWin_d * win(idx / n[d], n[d], m, sigma)
        κ_{d-1} = κ_d + (l_d-1) * N_d
        ζ_{d-1} = ζ_d + (gidx_d-1) * n_d
    end begin
        # bodyexpr
        I[κ_0,k] = ζ_0
        V[κ_0,k] = tmpWin_0
    end
  end
end

## precompute = FULL_LUT

@generated function _precomputeB(windowLUT::Vector, x, N::NTuple{D,Int}, n::NTuple{D,Int}, m, M, sigma, K, T, I,J,V,k) where D
  quote
    scale = T(1.0 / m * (K-1))
    @nexprs $D d->(xscale_d = x[d,k] * n[d])
    @nexprs $D d->(c_d = floor(Int, xscale_d))

    N_1 = 1
    @nexprs $D d->(N_{d+1} = N_d * (2*m+1))
    n_1 = 1
    @nexprs $D d->(n_{d+1} = n_d * n[d])

    @nexprs 1 d -> κ_{$D} = 1 # This is a hack, I actually want to write κ_$D = 1
    @nexprs 1 d -> ζ_{$D} = 1
    @nexprs 1 d -> tmpWin_{$D} = one(T)
    @nloops $D l d -> 1:(2*m+1) d->begin
        # preexpr
        off = c_d - m - 1
        gidx_d = rem(l_d + off + n[d], n[d]) + 1
        idx = abs( (xscale_d - l_d - off)*scale ) + 1
        idxL = floor(idx)
        idxInt = Int(idxL)
        tmpWin_{d-1} = tmpWin_d * (windowLUT[d][idxInt] + ( idx-idxL ) * (windowLUT[d][idxInt+1] - windowLUT[d][idxInt]))
        κ_{d-1} = κ_d + (l_d-1) * N_d
        ζ_{d-1} = ζ_d + (gidx_d-1) * n_d
    end begin
        # bodyexpr
        I[κ_0,k] = ζ_0
        V[κ_0,k] = tmpWin_0
    end
  end
end

function precomputeLUT(win, windowLUT, n, m, sigma, K, T)
    Z = round(Int, 3 * K / 2)
    for d = 1:length(windowLUT)
        windowLUT[d] = Vector{T}(undef, Z)
        @batch for l = 1:Z
            y = ((l - 1) / (K - 1)) * m / n[d]
            windowLUT[d][l] = win(y, n[d], m, sigma)
        end
    end
end

"""
precompute LUT for the multidimensional interpolation window
"""
function precomp_windowHatInvLUT(T::Type, win_hat::Function, N::NTuple{D,Int64}, sigma::Real, m::Int64) where D
    # size of oversampled grid
    n = ntuple(d->round(Int,sigma*N[d]), D)
    # lookup tables for 1d interpolation kernels
    windowHatInvLUT1d = Vector{Vector{T}}(undef,D)
    for d=1:D
        windowHatInvLUT1d[d] = [1.0/win_hat(k-1-N[d]/2, n[d], m, sigma) for k=1:N[d]]
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