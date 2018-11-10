# This is the original 1D implementation. Just for reference (not used anymore)

function apodization!(p::NFFTPlan{1,0}, f::AbstractVector{T}, g::StridedVector{T}) where T
    n = p.n[1]
    N = p.N[1]
    offset = round( Int, n - N / 2 ) - 1
    for l=1:N
        g[((l+offset)% n) + 1] = f[l] * p.windowHatInvLUT[1][l]
    end
end

function apodization_adjoint!(p::NFFTPlan{1,0}, g::AbstractVector{T}, f::StridedVector{T}) where T
    n = p.n[1]
    N = p.N[1]
    offset = round( Int, n - N / 2 ) - 1
    for l=1:N
        f[l] = g[((l+offset)% n) + 1] * p.windowHatInvLUT[1][l]
    end
end

function convolve!(p::NFFTPlan{1,0}, g::AbstractVector{T}, fHat::StridedVector{T}) where T
    fill!(fHat, zero(T))
    scale = 1.0 / p.m * (p.K-1)
    n = p.n[1]

    for k=1:p.M # loop over nonequispaced nodes
        c = floor(Int, p.x[k]*n)
        for l=(c-p.m):(c+p.m) # loop over nonzero elements
            gidx = rem(l+n, n) + 1
            idx = abs( (p.x[k]*n - l)*scale ) + 1
            idxL = floor(Int, idx)

            fHat[k] += g[gidx] * (p.windowLUT[1][idxL] + ( idx-idxL ) * (p.windowLUT[1][idxL+1] - p.windowLUT[1][idxL]))
        end
    end
end


function convolve_adjoint!(p::NFFTPlan{1,0}, fHat::AbstractVector{T}, g::StridedVector{T}) where T
    fill!(g, zero(T))
    scale = 1.0 / p.m * (p.K-1)
    n = p.n[1]

    for k=1:p.M # loop over nonequispaced nodes
        c = round(Int,p.x[k]*n)
        @inbounds @simd for l=(c-p.m):(c+p.m) # loop over nonzero elements
            gidx = rem(l+n, n) + 1
            idx = abs( (p.x[k]*n - l)*scale ) + 1
            idxL = round(Int, idx)

            g[gidx] += fHat[k] * (p.windowLUT[1][idxL] + ( idx-idxL ) * (p.windowLUT[1][idxL+1] - p.windowLUT[1][idxL]))
        end
    end
end
