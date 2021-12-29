@generated function apodization!(p::NFFTPlan{D,0,T}, f::AbstractArray{U,D}, g::StridedArray{Complex{T},D}) where {D,T,U}
    quote
        @nexprs $D d -> offset_d = round(Int, p.n[d] - p.N[d]/2) - 1

        @nloops $D l f d->(gidx_d = rem(l_d+offset_d, p.n[d]) + 1) begin
            v = @nref $D f l
            @nexprs $D d -> v *= p.windowHatInvLUT[d][l_d]
            (@nref $D g gidx) = v
        end
    end
end

@generated function apodization_adjoint!(p::NFFTPlan{D,0,T}, g::AbstractArray{Complex{T},D}, f::StridedArray{U,D}) where {D,T,U}
    quote
        @nexprs $D d -> offset_d = round(Int, p.n[d] - p.N[d]/2) - 1

        @nloops $D l f begin
            v = @nref $D g d -> rem(l_d+offset_d, p.n[d]) + 1
            @nexprs $D d -> v *= p.windowHatInvLUT[d][l_d]
            (@nref $D f l) = v
        end
    end
end

function convolve!(p::NFFTPlan{D,0,T}, g::AbstractArray{Complex{T},D}, fHat::StridedVector{U}) where {D,T,U}
  if isempty(p.B)
    convolve_LUT!(p, g, fHat)
  else
    convolve_sparse_matrix!(p, g, fHat)
  end
end

function convolve_LUT!(p::NFFTPlan{D,0,T}, g::AbstractArray{Complex{T},D}, fHat::StridedVector{U}) where {D,T,U}
    scale = 1.0 / p.m * (p.K-1)

    @cthreads for k in 1:p.M
        fHat[k] = _convolve_LUT(p, g, scale, k)
    end
end


@generated function _convolve_LUT(p::NFFTPlan{D,0,T}, g::AbstractArray{Complex{T},D}, scale, k) where {D,T}
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

function convolve_sparse_matrix!(p::NFFTPlan{D,0,T}, g::AbstractArray{Complex{T},D}, fHat::StridedVector{U}) where {D,T,U}
  fill!(fHat, zero(T))

  mul!(fHat, transpose(p.B), vec(g))
end

function convolve_adjoint!(p::NFFTPlan{D,0,T}, fHat::AbstractVector{U}, g::StridedArray{Complex{T},D}) where {D,T,U}
  if isempty(p.B)
    if NFFT._use_threads[]
      convolve_adjoint_LUT_MT!(p, fHat, g)
    else
      convolve_adjoint_LUT!(p, fHat, g)
    end
  else
    convolve_adjoint_sparse_matrix!(p, fHat, g)
  end
end

function convolve_adjoint_LUT_MT!(p::NFFTPlan{D,0,T}, fHat::AbstractVector{U}, g::StridedArray{Complex{T},D}) where {D,T,U}
  fill!(g, zero(T))
  scale = 1.0 / p.m * (p.K-1)
  @time g_tmp = Array{Complex{T}}(undef, size(g)..., Threads.nthreads())
  @time fill!(g_tmp, zero(T))

  @time @cthreads for k in 1:p.M
    _convolve_adjoint_LUT_MT!(p, fHat, view(g_tmp, :,:,:,Threads.threadid()), scale, k)
  end   

  @time for l=1:Threads.nthreads()
     g .+= g_tmp[l]
  end
  return g
end

@generated function _convolve_adjoint_LUT_MT!(p::NFFTPlan{D,0,T}, fHat::AbstractVector{U}, 
                            g_local::StridedArray{Complex{T},D}, scale, k) where {D,T,U}
  quote
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
            (@nref $D g_local gidx) += v
        end
    end
end


@generated function convolve_adjoint_LUT!(p::NFFTPlan{D,0,T}, fHat::AbstractVector{U}, g::StridedArray{Complex{T},D}) where {D,T,U}
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

function convolve_adjoint_sparse_matrix!(p::NFFTPlan{D,0,T},
                        fHat::AbstractVector{U}, g::StridedArray{Complex{T},D}) where {D,T,U}
  fill!(g, zero(T))

  mul!(vec(g), p.B, fHat)
end
