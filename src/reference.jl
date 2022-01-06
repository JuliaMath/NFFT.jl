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
    scale = 1.0 / p.m * (p.LUTSize-1)
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
    scale = 1.0 / p.m * (p.LUTSize-1)
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

# This is the old Cartesian implementations. Just for reference

@generated function apodization!(p::NFFTPlan{T,D,1}, f::AbstractArray{U,D}, g::StridedArray{Complex{T},D}) where {D,T,U}
    quote
        @nexprs $D d -> offset_d = round(Int, p.n[d] - p.N[d]/2) - 1

        @nloops $(D) l f d->(gidx_d = rem(l_d+offset_d, p.n[d]) + 1) begin
            v = @nref $D f l
            @nexprs $D d -> v *= p.windowHatInvLUT[d][l_d]
            (@nref $D g gidx) = v
        end
    end
end


@generated function apodization_adjoint!(p::NFFTPlan{T,D,1}, g::AbstractArray{Complex{T},D}, f::StridedArray{U,D}) where {D,T,U}
  quote
      @nexprs $D d -> offset_d = round(Int, p.n[d] - p.N[d]/2) - 1

      @inbounds @nloops $D l f begin
          v = @nref $D g d -> rem(l_d+offset_d, p.n[d]) + 1
          @nexprs $D d -> v *= p.windowHatInvLUT[d][l_d]
          (@nref $D f l) = v
      end
  end
end