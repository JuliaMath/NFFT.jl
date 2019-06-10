@generated function apodization!(p::NFFTPlan{D,0}, f::AbstractArray{<:Number,D}, g::StridedArray{<:Number,D}) where {D}
    quote
        @nexprs $D d -> offset_d = round(Int, p.n[d] - p.N[d]/2) - 1

        @nloops $D l f d->(gidx_d = rem(l_d+offset_d, p.n[d]) + 1) begin
            v = @nref $D f l
            @nexprs $D d -> v *= p.windowHatInvLUT[d][l_d]
            (@nref $D g gidx) = v
        end
    end
end

@generated function apodization_adjoint!(p::NFFTPlan{D,0}, g::AbstractArray{T,D}, f::StridedArray{T,D}) where {D,T}
    quote
        @nexprs $D d -> offset_d = round(Int, p.n[d] - p.N[d]/2) - 1

        @nloops $D l f begin
            v = @nref $D g d -> rem(l_d+offset_d, p.n[d]) + 1
            @nexprs $D d -> v *= p.windowHatInvLUT[d][l_d]
            (@nref $D f l) = v
        end
    end
end

function convolve!(p::NFFTPlan{D,0}, g::AbstractArray{<:Number,D}, fHat::StridedVector{<:Number}) where {D}
  if isempty(p.B)
    convolve_LUT!(p, g, fHat)
  else
    convolve_sparse_matrix!(p, g, fHat)
  end
end

function convolve_LUT!(p::NFFTPlan{D,0}, g::AbstractArray{T,D}, fHat::StridedVector{T}) where {D,T}
    scale = 1.0 / p.m * (p.K-1)

    #Threads.@threads
    for k in 1:p.M
        fHat[k] = _convolve_LUT(p, g, scale, k)
    end
end


@generated function _convolve_LUT(p::NFFTPlan{D,0}, g::AbstractArray{T,D}, scale, k) where {D,T}
    quote
        @nexprs $D d -> xscale_d = p.x[d,k] * p.n[d]
        @nexprs $D d -> c_d = floor(Int, xscale_d)

        fHat = zero(T)

        @inbounds @nloops $D l d -> (c_d-p.m):(c_d+p.m) d->begin
            # preexpr
            gidx_d = rem(l_d+p.n[d], p.n[d]) + 1
            idx = abs( (xscale_d - l_d)*scale ) + 1
            idxL = floor(idx)
            idxInt = Int(idxL)
            tmpWin_d = p.windowLUT[d][idxInt] + ( idx-idxL ) * (p.windowLUT[d][idxInt+1] - p.windowLUT[d][idxInt])
        end begin
            # bodyexpr
            v = @nref $D g gidx
            @nexprs $D d -> v *= tmpWin_d
            fHat += v
        end

        return fHat
    end
end

function convolve_sparse_matrix!(p::NFFTPlan{D,0}, g::AbstractArray{T,D}, fHat::StridedVector{T}) where {D,T}
  fill!(fHat, zero(T))

  mul!(fHat, transpose(p.B), vec(g))
end

function convolve_adjoint!(p::NFFTPlan{D,0}, fHat::AbstractVector{T}, g::StridedArray{T,D}) where {D,T}
  if isempty(p.B)
    convolve_adjoint_LUT!(p, fHat, g)
  else
    convolve_adjoint_sparse_matrix!(p, fHat, g)
  end
end

@generated function convolve_adjoint_LUT!(p::NFFTPlan{D,0}, fHat::AbstractVector{T}, g::StridedArray{T,D}) where {D,T}
    quote
        fill!(g, zero(T))
        scale = 1.0 / p.m * (p.K-1)

        @inbounds @simd for k in 1:p.M
            @nexprs $D d -> xscale_d = p.x[d,k] * p.n[d]
            @nexprs $D d -> c_d = floor(Int, xscale_d)

            @nloops $D l d -> (c_d-p.m):(c_d+p.m) d->begin
                # preexpr
                gidx_d = rem(l_d+p.n[d], p.n[d]) + 1
                idx = abs( (xscale_d - l_d)*scale ) + 1
                idxL = floor(idx)
                idxInt = Int(idxL)
                tmpWin_d = p.windowLUT[d][idxInt] + ( idx-idxL ) * (p.windowLUT[d][idxInt+1] - p.windowLUT[d][idxInt])
            end begin
                # bodyexpr
                v = fHat[k]
                @nexprs $D d -> v *= tmpWin_d
                (@nref $D g gidx) += v
            end
        end
    end
end

function convolve_adjoint_sparse_matrix!(p::NFFTPlan{D,0},
                        fHat::AbstractVector{T}, g::StridedArray{T,D}) where {D,T}
  fill!(g, zero(T))

  mul!(vec(g), p.B, fHat)
end
