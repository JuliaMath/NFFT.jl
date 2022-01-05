
function apodization!(p::NFFTPlan{T,D,R}, f::AbstractArray{U,D},
                      g::StridedArray{Complex{T},D}) where {T,D,R,U}
  d = Val(p.dims[1])
  return _apodization!(p, f, g, d)
end

@generated function _apodization!(p::NFFTPlan{T,D,R}, f::AbstractArray{U,D},
                                 g::StridedArray{Complex{T},D}, d::Val{DIM}) where {T,D,R,U,DIM}
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

function apodization_adjoint!(p::NFFTPlan{T,D,R},
                 g::AbstractArray{Complex{T},D}, f::StridedArray{U,D}) where {T,D,R,U}
  d = Val(p.dims[1])
  return _apodization_adjoint!(p, g, f, d)
end


@generated function _apodization_adjoint!(p::NFFTPlan{T,D,R},
           g::AbstractArray{Complex{T},D}, f::StridedArray{U,D}, d::Val{DIM}) where {T,D,R,U,DIM}
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

function convolve!(p::NFFTPlan{T,D,R}, g::AbstractArray{Complex{T},D},
             fHat::StridedArray{U,D}) where {T,D,R,U}
  d = Val(p.dims[1])
  return _convolve!(p, g, fHat, d)
end

@generated function _convolve!(p::NFFTPlan{T,D,R}, g::AbstractArray{Complex{T},D},
                              fHat::StridedArray{U,D}, d::Val{DIM}) where {T,D,R,U,DIM}
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

function convolve_adjoint!(p::NFFTPlan{T,D,R}, fHat::AbstractArray{U,D},
                 g::StridedArray{Complex{T},D}) where {T,D,R,U}
  d = Val(p.dims[1])
  return _convolve_adjoint!(p, fHat, g, d)
end


@generated function _convolve_adjoint!(p::NFFTPlan{T,D,R}, fHat::AbstractArray{U,D},
                                      g::StridedArray{Complex{T},D}, d::Val{DIM}) where {T,D,R,U,DIM}
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
