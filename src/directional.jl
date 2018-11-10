
@generated function apodization!(p::NFFTPlan{D,DIM}, f::AbstractArray{T,D}, g::StridedArray{T,D}) where {D,DIM,T}
    quote
        offset = round(Int, p.n[$DIM] - p.N[$DIM]/2) - 1

        @nloops $D l f d->begin
            # preexpr
            if d == $DIM
                gidx_d = rem(l_d+offset, p.n[d]) + 1
                winidx = l_d
            else
                gidx_d = l_d
            end
        end begin
            # bodyexpr
            (@nref $D g gidx) = (@nref $D f l) * p.windowHatInvLUT[1][winidx]
        end
    end
end

@generated function apodization_adjoint!(p::NFFTPlan{D,DIM}, g::AbstractArray{T,D}, f::StridedArray{T,D}) where {D,DIM,T}
    quote
        offset = round(Int, p.n[$DIM] - p.N[$DIM]/2) - 1

        @nloops $D l f d->begin
            # preexpr
            if d == $DIM
                gidx_d = rem(l_d+offset, p.n[d]) + 1
                winidx = l_d
            else
                gidx_d = l_d
            end
        end begin
            # bodyexpr
            (@nref $D f l) = (@nref $D g gidx) * p.windowHatInvLUT[1][winidx]
        end
    end
end



@generated function convolve!(p::NFFTPlan{D,DIM}, g::AbstractArray{T,D}, fHat::StridedArray{T,D}) where {D,DIM,T}
    quote
        fill!(fHat, zero(T))
        scale = 1.0 / p.m * (p.K-1)

        for k in 1:p.M
            xscale = p.x[k] * p.n[$DIM]
            c = floor(Int, xscale)
            @nloops $D l d->begin
                # rangeexpr
                if d == $DIM
                    (c-p.m):(c+p.m)
                else
                    1:size(g,d)
                end
            end d->begin
                # preexpr
                if d == $DIM
                    gidx_d = rem(l_d+p.n[d], p.n[d]) + 1
                    idx = abs( (xscale - l_d)*scale ) + 1
                    idxL = floor(idx)
                    idxInt = Int(idxL)
                    tmpWin = p.windowLUT[1][idxInt] + ( idx-idxL ) * (p.windowLUT[1][idxInt+1] - p.windowLUT[1][idxInt])
                    fidx_d = k
                else
                    gidx_d = l_d
                    fidx_d = l_d
                end
            end begin
                # bodyexpr
                (@nref $D fHat fidx) += (@nref $D g gidx) * tmpWin
            end
        end
    end
end


@generated function convolve_adjoint!(p::NFFTPlan{D,DIM}, fHat::AbstractArray{T,D}, g::StridedArray{T,D}) where {D,DIM,T}
    quote
        fill!(g, zero(T))
        scale = 1.0 / p.m * (p.K-1)

        for k in 1:p.M
            xscale = p.x[k] * p.n[$DIM]
            c = floor(Int, xscale)
            @nloops $D l d->begin
                # rangeexpr
                if d == $DIM
                    (c-p.m):(c+p.m)
                else
                    1:size(g,d)
                end
            end d->begin
                # preexpr
                if d == $DIM
                    gidx_d = rem(l_d+p.n[d], p.n[d]) + 1
                    idx = abs( (xscale - l_d)*scale ) + 1
                    idxL = floor(idx)
                    idxInt = Int(idxL)
                    tmpWin = p.windowLUT[1][idxInt] + ( idx-idxL ) * (p.windowLUT[1][idxInt+1] - p.windowLUT[1][idxInt])
                    fidx_d = k
                else
                    gidx_d = l_d
                    fidx_d = l_d
                end
            end begin
                # bodyexpr
                (@nref $D g gidx) += (@nref $D fHat fidx) * tmpWin
            end
        end
    end
end
