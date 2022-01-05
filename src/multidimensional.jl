function apodization!(p::NFFTPlan{T,1,1}, f::AbstractVector{U}, g::StridedVector{Complex{T}}) where {T,U}
  n = p.n[1]
  N = p.N[1]
  N2 = N÷2
  offset = n - N÷2
  @cthreads for l=1:N÷2
      g[l+offset] = f[l] * p.windowHatInvLUT[1][l]
      g[l] = f[l+N2] * p.windowHatInvLUT[1][l+N2]
  end
end

function apodization!(p::NFFTPlan{T,D,1}, f::AbstractArray{U,D}, g::StridedArray{Complex{T},D}) where {D,T,U}
  @cthreads for o = 1:p.N[end]
      _apodization!(p, f, g, o)  
  end
end

@generated function _apodization!(p::NFFTPlan{T,D,1}, f::AbstractArray{U,D}, g::StridedArray{Complex{T},D}, o) where {D,T,U}
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

function apodization_adjoint_1D!(p::NFFTPlan{T,1,1}, g::StridedVector{Complex{T}}, f::AbstractVector{U}) where {T,U}
  n = p.n[1]
  N = p.N[1]
  N2 = N÷2
  offset = n - N÷2
  @cthreads for l=1:N÷2
      f[l] =  g[l+offset] * p.windowHatInvLUT[1][l]
      f[l+N2] = g[l] * p.windowHatInvLUT[1][l+N2]
  end
end

function apodization_adjoint!(p::NFFTPlan{T,D,1}, g::AbstractArray{Complex{T},D}, f::StridedArray{U,D}) where {D,T,U}
  if D == 1
    apodization_adjoint_1D!(p, g, f)
  else
    @cthreads for o = 1:p.N[end]
      _apodization_adjoint!(p, g, f, o)  
    end
  end
end

@generated function _apodization_adjoint!(p::NFFTPlan{T,D,1}, g::AbstractArray{Complex{T},D}, f::StridedArray{U,D}, o) where {D,T,U}
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

function convolve!(p::NFFTPlan{T,D,1}, g::AbstractArray{Complex{T},D}, fHat::StridedVector{U}) where {D,T,U}
  if isempty(p.B)
    convolve_LUT!(p, g, fHat)
  else
    convolve_sparse_matrix!(p, g, fHat)
  end
end

function convolve_LUT!(p::NFFTPlan{T,D,1}, g::AbstractArray{Complex{T},D}, fHat::StridedVector{U}) where {D,T,U}
  L = Val(2*p.m+1)
  scale = T(1.0 / p.m * (p.K-1))

  @cthreads for k in 1:p.M
      fHat[k] = _convolve_LUT(p, g, L, scale, k)
  end
end

function _precomputeOneNode(p::NFFTPlan{T,D,1}, scale, k, d, L::Val{Z}) where {T,D,Z}
    return _precomputeOneNode(p.windowLUT, p.x, p.n, p.m, p.sigma, scale, k, d, L) 
end

@generated function _convolve_LUT(p::NFFTPlan{T,D,1}, g::AbstractArray{Complex{T},D}, L::Val{Z}, scale, k) where {D,T,Z}
  quote
    @nexprs $(D) d -> ((tmpIdx_d, tmpWin_d) = _precomputeOneNode(p, scale, k, d, L) )
  
    fHat = zero(Complex{T})

    @nexprs 1 d -> prodWin_{$D} = one(T)
    @nloops_ $D l d -> 1:$Z d->begin
      # preexpr
      prodWin_{d-1} = prodWin_d * tmpWin_d[l_d]
      gidx_d = tmpIdx_d[l_d] 
    end begin
      # bodyexpr
      fHat += prodWin_0 * @nref $D g gidx
    end
    return fHat
  end
end

function convolve_sparse_matrix!(p::NFFTPlan{T,D,1}, g::AbstractArray{Complex{T},D}, fHat::StridedVector{U}) where {D,T,U}
  threaded_mul!(fHat, transpose(p.B), vec(g))
end

function convolve_adjoint!(p::NFFTPlan{T,D,1}, fHat::AbstractVector{U}, g::StridedArray{Complex{T},D}) where {D,T,U}
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

#=function convolve_adjoint_LUT_MT!(p::NFFTPlan{T,D,1}, fHat::AbstractVector{U}, g::StridedArray{Complex{T},D}) where {D,T,U}
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
end=#


function convolve_adjoint_LUT!(p::NFFTPlan{T,D,1}, fHat::AbstractVector{U}, g::StridedArray{Complex{T},D}) where {D,T,U}
  fill!(g, zero(T))
  L = Val(2*p.m+1)
  scale = T(1.0 / p.m * (p.K-1))

  @inbounds @simd for k in 1:p.M
    _convolve_adjoint_LUT!(p, fHat, g, L, scale, k)
  end
end

@generated function _convolve_adjoint_LUT!(p::NFFTPlan{T,D,1}, fHat::AbstractVector{U}, g::StridedArray{Complex{T},D}, L::Val{Z}, scale, k) where {D,T,U,Z}
  quote
    @nexprs $(D) d -> ((tmpIdx_d, tmpWin_d) = _precomputeOneNode(p, scale, k, d, L) )

    @nexprs 1 d -> prodWin_{$D} = one(T)
    @nloops_ $D l d -> 1:$Z d->begin
      # preexpr
      prodWin_{d-1} = prodWin_d * tmpWin_d[l_d]
      gidx_d = tmpIdx_d[l_d] 
    end begin
      # bodyexpr
      (@nref $D g gidx) += prodWin_0 * fHat[k] 
    end
  end
end

function convolve_adjoint_sparse_matrix!(p::NFFTPlan{T,D,1},
                        fHat::AbstractVector{U}, g::StridedArray{Complex{T},D}) where {D,T,U}
  threaded_mul!(vec(g), p.B, fHat)
  #mul!(vec(g), p.B, fHat)
end

