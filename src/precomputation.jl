
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