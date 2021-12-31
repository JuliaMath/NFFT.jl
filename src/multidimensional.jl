function apodization!(p::NFFTPlan{1,0,T}, f::AbstractVector{U}, g::StridedVector{Complex{T}}) where {T,U}
  n = p.n[1]
  N = p.N[1]
  N2 = N÷2
  offset = n - N÷2
  @cthreads for l=1:N÷2
      g[l+offset] = f[l] * p.windowHatInvLUT[1][l]
      g[l] = f[l+N2] * p.windowHatInvLUT[1][l+N2]
  end
end

function apodization!(p::NFFTPlan{D,0,T}, f::AbstractArray{U,D}, g::StridedArray{Complex{T},D}) where {D,T,U}
  @cthreads for o = 1:p.N[end]
      _apodization!(p, f, g, o)  
  end
end

@generated function _apodization!(p::NFFTPlan{D,0,T}, f::AbstractArray{U,D}, g::StridedArray{Complex{T},D}, o) where {D,T,U}
  quote
    oo = rem(o+p.n[$D] - p.N[$D]÷ 2 - 1, p.n[$D]) + 1
    @nloops $(D-2) l d->(1:size(f,d+1)) d->(gidx_d = rem(l_d+p.n[d+1] - p.N[d+1]÷2 - 1, p.n[d+1]) + 1) begin
      N2 = p.N[1]÷2
      @inbounds @simd for i = 1:N2
        v = f[i, CartesianIndex(@ntuple $(D-2) l), o]
        v *= p.windowHatInvLUT[1][i] * p.windowHatInvLUT[$D][o]
        @nexprs $(D-2) d -> v *= p.windowHatInvLUT[d+1][l_d]
        g[i-N2+p.n[1], CartesianIndex(@ntuple $(D-2) gidx), oo] = v

        v = f[i+N2, CartesianIndex(@ntuple $(D-2) l), o] 
        v *= p.windowHatInvLUT[1][i+N2] * p.windowHatInvLUT[$D][o]
        @nexprs $(D-2) d -> v *= p.windowHatInvLUT[d+1][l_d]
        g[i, CartesianIndex(@ntuple $(D-2) gidx), oo] = v
      end
    end
  end
end

function apodization_adjoint_1D!(p::NFFTPlan{1,0,T}, g::StridedVector{Complex{T}}, f::AbstractVector{U}) where {T,U}
  n = p.n[1]
  N = p.N[1]
  N2 = N÷2
  offset = n - N÷2
  @cthreads for l=1:N÷2
      f[l] =  g[l+offset] * p.windowHatInvLUT[1][l]
      f[l+N2] = g[l] * p.windowHatInvLUT[1][l+N2]
  end
end

function apodization_adjoint!(p::NFFTPlan{D,0,T}, g::AbstractArray{Complex{T},D}, f::StridedArray{U,D}) where {D,T,U}
  if D == 1
    apodization_adjoint_1D!(p, g, f)
  else
    @cthreads for o = 1:p.N[end]
      _apodization_adjoint!(p, g, f, o)  
    end
  end
end

@generated function _apodization_adjoint!(p::NFFTPlan{D,0,T}, g::AbstractArray{Complex{T},D}, f::StridedArray{U,D}, o) where {D,T,U}
  quote
    oo = rem(o+p.n[$D] - p.N[$D]÷ 2 - 1, p.n[$D]) + 1
    @nloops $(D-2) l d->(1:size(f,d+1)) d->(gidx_d = rem(l_d+p.n[d+1] - p.N[d+1]÷2 - 1, p.n[d+1]) + 1) begin
      N2 = p.N[1]÷2
      for i = 1:N2
        v = g[i-N2+p.n[1], CartesianIndex(@ntuple $(D-2) gidx), oo] 
        v *= p.windowHatInvLUT[1][i] * p.windowHatInvLUT[$D][o]
        @nexprs $(D-2) d -> v *= p.windowHatInvLUT[d+1][l_d]
        f[i, CartesianIndex(@ntuple $(D-2) l), o] = v

        v = g[i, CartesianIndex(@ntuple $(D-2) gidx), oo] 
        v *= p.windowHatInvLUT[1][i+N2] * p.windowHatInvLUT[$D][o]
        @nexprs $(D-2) d -> v *= p.windowHatInvLUT[d+1][l_d]
        f[i+N2, CartesianIndex(@ntuple $(D-2) l), o] = v
      end
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
    scale = T(1.0 / p.m * (p.K-1))

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

  bmul!(fHat, SparseMatrixCSR(transpose(p.B)), vec(g))
  #mul!(fHat, transpose(p.B), vec(g))
end

function convolve_adjoint!(p::NFFTPlan{D,0,T}, fHat::AbstractVector{U}, g::StridedArray{Complex{T},D}) where {D,T,U}
  if isempty(p.B)
    #if NFFT._use_threads[]
    #  convolve_adjoint_LUT_MT!(p, fHat, g)
    #else
      convolve_adjoint_LUT!(p, fHat, g)
    #end
  else
    convolve_adjoint_sparse_matrix!(p, fHat, g)
  end
end

function convolve_adjoint_LUT_MT!(p::NFFTPlan{D,0,T}, fHat::AbstractVector{U}, g::StridedArray{Complex{T},D}) where {D,T,U}
  fill!(g, zero(T))
  scale = T(1.0 / p.m * (p.K-1))
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
                            g_local, scale, k) where {D,T,U}
  quote
        @nexprs $D d -> xscale_d = p.x[d,k] * p.n[d]
        @nexprs $D d -> c_d = floor(Int, xscale_d)

        @inbounds @nloops $D l d -> (c_d-p.m):(c_d+p.m) d->begin
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
      scale = T(1.0 / p.m * (p.K-1))

      @inbounds @simd  for k in 1:p.M
          @nexprs $D d -> xscale_d = p.x[d,k] * p.n[d]
          @nexprs $D d -> c_d = floor(Int, xscale_d)

          @nloops_ $D l d -> (c_d-p.m):(c_d+p.m) d->begin
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
