
@generated function precomputeB(win, x, n::NTuple{D,Int}, m, M, sigma, T, U1, U2) where D
    quote
        I = zeros(Int64, (2*m+1)^D, M)
        J = zeros(Int64, M+1)
        V = zeros(T, (2*m+1)^D, M)

        J[1] = 1
        @inbounds @simd for k in 1:M
            @nexprs $D d -> xscale_d = x[d,k] * n[d]
            @nexprs $D d -> c_d = floor(Int, xscale_d)

            @nloops $D l d -> (c_d-m):(c_d+m) d->begin
                # preexpr
                gidx_d = rem(l_d+n[d], n[d]) + 1
                Iidx_d = l_d - c_d + m + 1
                idx = abs( (xscale_d - l_d) )
                tmpWin_d =  win(idx / n[d], n[d], m, sigma)

            end begin
                # bodyexpr
                v = 1
                @nexprs $D d -> v *= tmpWin_d
                i1 = 1
                @nexprs $D d -> i1 += (Iidx_d-1) * U1[d]
                i2 = 1
                @nexprs $D d -> i2 += (gidx_d-1) * U2[d]

                I[i1,k] = i2
                V[i1,k] = v
            end
            J[k+1] = J[k] + (2*m+1)^D
        end

        S = SparseMatrixCSC(prod(n), M, J, vec(I), vec(V))
        return S
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
function precomp_apodIdx(N::NTuple{1,Int64}, sigma::Real, numSlices=1)
    # size of oversampled grid
    n = ntuple(d->round(Int,sigma*N[d]), 1)
    # offsets to centran NxN-region
    offset = ntuple(d->round(Int, n[d]-N[d]/2)-1, 1)
    # linear Indices
    linIdx = LinearIndices((n[1],numSlices))
    # apodization indices
    apodIdx = zeros(Int64,N[1],numSlices)
    for i_1=1:N[1]
        idx1 = rem(i_1+offset[1], n[1]) + 1
        # store linearIdx
        apodIdx[i_1,:] .= linIdx[idx1,:]
    end
  
    return vec(apodIdx)
end
  
function precomp_apodIdx(N::NTuple{2,Int64}, sigma::Real, numSlices=1)
    # size of oversampled grid
    n = ntuple(d->round(Int,sigma*N[d]), 2)
    # offsets to centran NxN-region
    offset = ntuple(d->round(Int, n[d]-N[d]/2)-1, 2)
    # linear Indices
    linIdx = LinearIndices((n[1],n[2],numSlices))
    # apodization indices
    apodIdx = zeros(Int64,N[1],N[2],numSlices)
    for i_2=1:N[2]
        idx2 = rem(i_2+offset[2], n[2]) + 1
        for i_1=1:N[1]
          idx1 = rem(i_1+offset[1], n[1]) + 1
          # store linearIdx
          apodIdx[i_1,i_2,:] .= linIdx[idx1,idx2,:]
        end
    end
  
    return vec(apodIdx)
end
  
function precomp_apodIdx(N::NTuple{3,Int64}, sigma::Real, numSlices=1)
    # size of oversampled grid
    n = ntuple(d->round(Int,sigma*N[d]), 3)
    # offsets to centran NxN-region
    offset = ntuple(d->round(Int, n[d]-N[d]/2)-1, 3)
    # linear Indices
    linIdx = LinearIndices((n[1],n[2],n[3],numSlices))
    # apodization indices
    apodIdx = zeros(Int64,N[1],N[2],N[3],numSlices)
    for i_3=1:N[3]
        idx3 = rem(i_3+offset[3], n[3]) + 1
        for i_2=1:N[2]
            idx2 = rem(i_2+offset[2], n[2]) + 1
            for i_1=1:N[1]
                idx1 = rem(i_1+offset[1], n[1]) + 1
                # store linearIdx
                apodIdx[i_1,i_2,i_3,:] .= linIdx[idx1,idx2,idx3,:]
            end
        end
    end
  
    return vec(apodIdx)
end