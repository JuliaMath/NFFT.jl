module NFFT

import Base.ind2sub
using Base.Cartesian

export NFFTPlan, nfft, nfft_adjoint, ndft, ndft_adjoint, nfft_performance,
       sdc

# Some internal documentation (especially for people familiar with the nfft)
#
# - Currently the window cannot be changed and defaults to the kaiser-bessel
#   window. This is done for simplicity and due to the fact that the 
#   kaiser-bessel window usually outperforms any other window
#
# - The window is precomputed during construction of the NFFT plan
#   When performing the nfft convolution, the LUT of the window is used to
#   perform linear interpolation. This approach is reasonable fast and does not
#   require too much memory. There are, however alternatives known that are either 
#   faster or require no extra memory at all.

function window_kaiser_bessel(x,n,m,sigma)
  b = pi*(2-1/sigma)
  arg = m^2-n^2*x^2
  if abs(x) < m/n
    y = sinh(b*sqrt(arg))/sqrt(arg)/pi
  elseif abs(x) > m/n
    y = zero(x)
  else
    y = b/pi
  end
  return y
end

function window_kaiser_bessel_hat(k,n,m,sigma)
  b = pi*(2-1/sigma)
  return besseli(0,m*sqrt(b^2-(2*pi*k/n)^2))
end

type NFFTPlan{D,T}
  N::NTuple{D,Int64}
  M::Int64
  x::Array{T,2}
  m::Int64
  sigma::T
  n::NTuple{D,Int64}
  K::Int64
  windowLUT::Vector{Vector{T}}
  windowHatInvLUT::Vector{Vector{T}}
  tmpVec::Array{Complex{T},D}
end

function NFFTPlan{D,T}(x::Array{T,2}, N::NTuple{D,Int}, m=4, sigma=2.0, K=2000)
  
  if D != size(x,1)
    throw(ArgumentError())
  end

  n = ntuple(d->round(Int,sigma*N[d]), D)

  tmpVec = zeros(Complex{T}, n)

  M = size(x,2)

  # Create lookup table
  
  windowLUT = Array(Vector{T},D)
  for d=1:D
    Z = round(Int,3*K/2)
    windowLUT[d] = zeros(T, Z)
    for l=1:Z
      y = ((l-1) / (K-1)) * m/n[d]
      windowLUT[d][l] = window_kaiser_bessel(y, n[d], m, sigma)
    end
  end

  windowHatInvLUT = Array(Vector{T}, D)
  for d=1:D
    windowHatInvLUT[d] = zeros(T, N[d])
    for k=1:N[d]
      windowHatInvLUT[d][k] = 1. / window_kaiser_bessel_hat(k-1-N[d]/2, n[d], m, sigma)
    end
  end

  NFFTPlan(N, M, x, m, sigma, n, K, windowLUT, windowHatInvLUT, tmpVec )
end

function NFFTPlan{T}(x::Array{T,1}, N::Integer, m=4, sigma=2.0)
  NFFTPlan(reshape(x,1,length(x)), (N,), m, sigma)
end

function consistencyCheck{T,D}(p::NFFTPlan{D}, f::AbstractArray{T,D}, fHat::AbstractVector{T})
	if p.N != size(f) || p.M != length(fHat)
		throw(DimensionMismatch("Data is not consistent with NFFTPlan"))
	end
end

### nfft functions ###

function nfft!{T,D}(p::NFFTPlan{D}, f::AbstractArray{T,D}, fHat::StridedVector{T})
  consistencyCheck(p, f, fHat)

  fill!(p.tmpVec, zero(T))
  @inbounds apodization!(p, f, p.tmpVec)
  fft!(p.tmpVec)
  fill!(fHat, zero(T))
  @inbounds convolve!(p, p.tmpVec, fHat)
  return fHat
end

function nfft{T,D}(p::NFFTPlan, f::AbstractArray{T,D})
  fHat = zeros(T, p.M)
  nfft!(p, f, fHat)
  return fHat
end

function nfft{T,D}(x, f::AbstractArray{T,D})
  p = NFFTPlan(x, size(f) )
  return nfft(p, f)
end

function nfft_adjoint!{T,D}(p::NFFTPlan{D}, fHat::AbstractArray{T}, f::StridedArray{T,D})
  consistencyCheck(p, f, fHat)

  fill!(p.tmpVec, zero(T))
  @inbounds convolve_adjoint!(p, fHat, p.tmpVec)
  ifft!(p.tmpVec)
  scale!(p.tmpVec, prod(p.n))
  fill!(f, zero(T))
  @inbounds apodization_adjoint!(p, p.tmpVec, f)
  return f
end

function nfft_adjoint{T,D}(p::NFFTPlan{D}, fHat::Vector{Complex{T}})
  f = zeros(Complex{T},p.N)
  nfft_adjoint!(p, fHat, f)
  return f
end

function nfft_adjoint{T,D}(x, N::NTuple{D,Int}, fHat::Vector{T})
  p = NFFTPlan(x, N)
  return nfft_adjoint(p, fHat)
end

### ndft functions ###

# fallback for 1D 
function ind2sub{T}(::Array{T,1}, idx::Int)
  idx
end

function ndft{T,D}(plan::NFFTPlan{D}, f::AbstractArray{T,D})
  plan.N == size(f) || throw(DimensionMismatch("Data is not consistent with NFFTPlan"))

  g = zeros(T, plan.M)

  for l=1:prod(plan.N)
    idx = ind2sub(plan.N,l)

    for k=1:plan.M
      arg = zero(T)
      for d=1:D
        arg += plan.x[d,k] * ( idx[d] - 1 - plan.N[d] / 2 )
      end
      g[k] += f[l] * exp(-2*pi*1im*arg)
    end
  end

  return g
end

function ndft_adjoint{T,D}(plan::NFFTPlan{D}, fHat::AbstractArray{T,1})
  plan.M == length(fHat) || throw(DimensionMismatch("Data is not consistent with NFFTPlan"))

  g = zeros(T, plan.N)

  for l=1:prod(plan.N)
    idx = ind2sub(plan.N,l)

    for k=1:plan.M
      arg = zero(T)
      for d=1:D
        arg += plan.x[d,k] * ( idx[d] - 1 - plan.N[d] / 2 )
      end
      g[l] += fHat[k] * exp(2*pi*1im*arg)
    end
  end

  return g
end



### convolve! ###

function convolve!{T}(p::NFFTPlan{1}, g::AbstractVector{T}, fHat::StridedVector{T})
  n = p.n[1]

  for k=1:p.M # loop over nonequispaced nodes
    c = floor(Int,p.x[k]*n)
    for l=(c-p.m):(c+p.m) # loop over nonzero elements

      idx = ((l+n)% n) + 1
      idx2 = abs(((p.x[k]*n - l)/p.m )*(p.K-1)) + 1
      idx2L = floor(Int,idx2)

      fHat[k] += g[idx] * (p.windowLUT[1][idx2L] + ( idx2-idx2L ) * (p.windowLUT[1][idx2L+1] - p.windowLUT[1][idx2L] ) )
    end
  end
end

function convolve!{T}(p::NFFTPlan{2}, g::AbstractMatrix{T}, fHat::StridedVector{T})
  scale = 1.0 / p.m * (p.K-1)

  n1 = p.n[1]
  n2 = p.n[2]

  for k=1:p.M # loop over nonequispaced nodes
    c0 = floor(Int, p.x[1,k]*n1)
    c1 = floor(Int, p.x[2,k]*n2)

    for l1=(c1-p.m):(c1+p.m) # loop over nonzero elements

      idx1 = ((l1+n2)% n2) + 1

      idx2 = abs((p.x[2,k]*n2 - l1)*scale) + 1
      idx2L = floor(Int,idx2)

      tmpWin = (p.windowLUT[2][idx2L] + ( idx2-idx2L ) * (p.windowLUT[2][idx2L+1] - p.windowLUT[2][idx2L] ) )

      for l0=(c0-p.m):(c0+p.m)

        idx0 = ((l0+n1)% n1) + 1
        idx2 = abs((p.x[1,k]*n1 - l0)*scale) + 1
        idx2L = round(Int,idx2)

        fHat[k] += g[idx0,idx1] * tmpWin * (p.windowLUT[1][idx2L] + ( idx2-idx2L ) * (p.windowLUT[1][idx2L+1] - p.windowLUT[1][idx2L] ) )
      end
    end
  end
end

function convolve!{T}(p::NFFTPlan{3}, g::AbstractArray{T,3}, fHat::StridedVector{T})
  scale = 1.0 / p.m * (p.K-1)

  n1 = p.n[1]
  n2 = p.n[2]
  n3 = p.n[3]

  for k=1:p.M # loop over nonequispaced nodes
    c0 = floor(Int,p.x[1,k]*n1)
    c1 = floor(Int,p.x[2,k]*n2)
    c2 = floor(Int,p.x[3,k]*n3)

    for l2=(c2-p.m):(c2+p.m) # loop over nonzero elements

      idx2 = ((l2+n3)% n3) + 1

      idxb = abs((p.x[3,k]*n3 - l2)*scale) + 1
      idxbL = floor(Int,idxb)

      tmpWin2 = (p.windowLUT[3][idxbL] + ( idxb-idxbL ) * (p.windowLUT[3][idxbL+1] - p.windowLUT[3][idxbL] ) )

      for l1=(c1-p.m):(c1+p.m)

        idx1 = ((l1+n2)% n2) + 1
        idxb = abs((p.x[2,k]*n2 - l1)*scale) + 1
        idxbL = floor(Int,idxb)

        tmpWin = (p.windowLUT[2][idxbL] + ( idxb-idxbL ) * (p.windowLUT[2][idxbL+1] - p.windowLUT[2][idxbL] ) )

        for l0=(c0-p.m):(c0+p.m)

          idx0 = ((l0+n1)% n1) + 1
          idxb = abs((p.x[1,k]*n1 - l0)*scale) + 1
          idxbL = round(Int,idxb)

          tmp = g[idx0,idx1,idx2]
          fHat[k] += tmp * tmpWin * tmpWin2 * (p.windowLUT[1][idxbL] + ( idxb-idxbL ) * (p.windowLUT[1][idxbL+1] - p.windowLUT[1][idxbL] ) )
        end
      end
    end
  end
end


function convolve!{T,D}(p::NFFTPlan{D}, g::AbstractArray{T,D}, fHat::StridedVector{T})
  l = Array(Int,D)
  idx = Array(Int,D)
  P = Array(Int,D)
  c = Array(Int,D)

  for k=1:p.M # loop over nonequispaced nodes

    for d=1:D
      c[d] = floor(Int,p.x[d,k]*p.n[d])
      P[d] = 2*p.m + 1
    end

    for j=1:prod(P) # loop over nonzero elements
      it = ind2sub(tuple(P...),j)
      for d=1:D
        l[d] = c[d]-p.m+it[d]
        idx[d] = ((l[d]+p.n[d])% p.n[d]) + 1
      end

      tmp = g[idx...]
      for d=1:D
        idx2 = abs(((p.x[d,k]*p.n[d] - l[d])/p.m )*(p.K-1)) + 1
        idx2L = floor(Int,idx2)
        tmp *= (p.windowLUT[d][idx2L] + ( idx2-idx2L ) * (p.windowLUT[d][idx2L+1] - p.windowLUT[d][idx2L] ) )
      end

      fHat[k] += tmp;
    end
  end
end


### convolve_adjoint! ###

function convolve_adjoint!{T}(p::NFFTPlan{1}, fHat::AbstractVector{T}, g::StridedVector{T})
  n = p.n[1]

  for k=1:p.M # loop over nonequispaced nodes
    c = round(Int,p.x[k]*n)
    for l=(c-p.m):(c+p.m) # loop over nonzero elements

      idx = ((l+n)%n)+1
      idx2 = abs(((p.x[k]*n - l)/p.m )*(p.K-1)) + 1
      idx2L = round(Int,idx2)

      g[idx] += fHat[k] * (p.windowLUT[1][idx2L] + ( idx2-idx2L ) * (p.windowLUT[1][idx2L+1] - p.windowLUT[1][idx2L] ) )
    end
  end
end

function convolve_adjoint!{T}(p::NFFTPlan{2}, fHat::AbstractVector{T}, g::StridedMatrix{T})
  scale = 1.0 / p.m * (p.K-1)
  n1 = p.n[1]
  n2 = p.n[2]

  for k=1:p.M # loop over nonequispaced nodes
    c0 = floor(Int,p.x[1,k]*n1)
    c1 = floor(Int,p.x[2,k]*n2)

    for l1=(c1-p.m):(c1+p.m) # loop over nonzero elements
      idx1 = ((l1+n2)%n2) + 1
      idx2 = abs((p.x[2,k]*n2 - l1)*scale) + 1
      idx2L = floor(Int,idx2)

      tmp = fHat[k] * (p.windowLUT[2][idx2L] + ( idx2-idx2L ) * (p.windowLUT[2][idx2L+1] - p.windowLUT[2][idx2L] ) )

      for l0=(c0-p.m):(c0+p.m)
        idx0 = ((l0+n1)%n1) + 1
        idx2 = abs((p.x[1,k]*n1 - l0)*scale) + 1
        idx2L = round(Int,idx2)
        g[idx0,idx1] += tmp * (p.windowLUT[1][idx2L] + ( idx2-idx2L ) * (p.windowLUT[1][idx2L+1] - p.windowLUT[1][idx2L] ) )
      end
    end
  end
end

function convolve_adjoint!{T}(p::NFFTPlan{3}, fHat::AbstractVector{T}, g::StridedArray{T,3})
  scale = 1.0 / p.m * (p.K-1)
  n1 = p.n[1]
  n2 = p.n[2]
  n3 = p.n[3]

  for k=1:p.M # loop over nonequispaced nodes
    c0 = floor(Int,p.x[1,k]*n1)
    c1 = floor(Int,p.x[2,k]*n2)
    c2 = floor(Int,p.x[3,k]*n3)

    for l2=(c2-p.m):(c2+p.m) # loop over nonzero elements
      idx2 = ((l2+n3)%n3) + 1
      idxb = abs((p.x[3,k]*n3 - l2)*scale) + 1
      idxbL = floor(Int,idxb)

      tmp = fHat[k] * (p.windowLUT[3][idxbL] + ( idxb-idxbL ) * (p.windowLUT[3][idxbL+1] - p.windowLUT[3][idxbL] ) )

      for l1=(c1-p.m):(c1+p.m)
        idx1 = ((l1+n2)%n2) + 1
        idxb = abs((p.x[2,k]*n2 - l1)*scale) + 1
        idxbL = floor(Int,idxb)

        tmp2 = tmp * (p.windowLUT[2][idxbL] + ( idxb-idxbL ) * (p.windowLUT[2][idxbL+1] - p.windowLUT[2][idxbL] ) )

        for l0=(c0-p.m):(c0+p.m)
          idx0 = ((l0+n1)%n1) + 1
          idxb = abs((p.x[1,k]*n1 - l0)*scale) + 1
          idxbL = round(Int,idxb)
          g[idx0,idx1,idx2] += tmp2 * (p.windowLUT[1][idxbL] + ( idxb-idxbL ) * (p.windowLUT[1][idxbL+1] - p.windowLUT[1][idxbL] ) )
        end
      end
    end
  end
end


function convolve_adjoint!{T,D}(p::NFFTPlan{D}, fHat::AbstractVector{T}, g::StridedArray{T,D})
  l = Array(Int,D)
  idx = Array(Int,D)
  P = Array(Int,D)
  c = Array(Int,D)

  for k=1:p.M # loop over nonequispaced nodes

    for d=1:D
      c[d] = floor(Int,p.x[d,k]*p.n[d])
      P[d] = 2*p.m + 1
    end

    for j=1:prod(P) # loop over nonzero elements
      it = ind2sub(tuple(P...),j)
      for d=1:D
        l[d] = c[d]-p.m+it[d]
        idx[d] = ((l[d]+p.n[d])%p.n[d]) + 1
      end

      tmp = fHat[k]
      for d=1:D
        idx2 = abs(((p.x[d,k]*p.n[d] - l[d])/p.m )*(p.K-1)) + 1
        idx2L = floor(Int,idx2)
        tmp *= (p.windowLUT[d][idx2L] + ( idx2-idx2L ) * (p.windowLUT[d][idx2L+1] - p.windowLUT[d][idx2L] ) )
      end

      g[idx...] += tmp;
    end
  end
end


### apodization! ###

function apodization!{T}(p::NFFTPlan{1}, f::AbstractVector{T}, g::StridedVector{T})
  n = p.n[1]
  N = p.N[1]
  const offset = round( Int, n - N / 2 ) - 1
  for l=1:N
    g[((l+offset)% n) + 1] = f[l] * p.windowHatInvLUT[1][l]
  end
end

function apodization!{T}(p::NFFTPlan{2}, f::AbstractMatrix{T}, g::StridedMatrix{T})
  n1 = p.n[1]
  N1 = p.N[1]
  n2 = p.n[2]
  N2 = p.N[2]
  const offset1 = round( Int, n1 - N1 / 2 ) - 1
  const offset2 = round( Int, n2 - N2 / 2 ) - 1
  for ly=1:N2
    for lx=1:N1
      g[((lx+offset1)% n1) + 1, ((ly+offset2)% n2) + 1] = f[lx, ly]  *  p.windowHatInvLUT[1][lx] * p.windowHatInvLUT[2][ly]
    end
  end
end

function apodization!{T}(p::NFFTPlan{3}, f::AbstractArray{T,3}, g::StridedArray{T,3})
  n1 = p.n[1]
  N1 = p.N[1]
  n2 = p.n[2]
  N2 = p.N[2]
  n3 = p.n[3]
  N3 = p.N[3]

  const offset1 = round( Int, n1 - N1 / 2 ) - 1
  const offset2 = round( Int, n2 - N2 / 2 ) - 1
  const offset3 = round( Int, n3 - N3 / 2 ) - 1
  for lz=1:N3
    for ly=1:N2
      for lx=1:N1
        g[((lx+offset1)% n1) + 1, ((ly+offset2)% n2) + 1, ((lz+offset3)% n3) + 1] = f[lx, ly, lz]  *  p.windowHatInvLUT[1][lx] * p.windowHatInvLUT[2][ly] * p.windowHatInvLUT[3][lz]
      end
    end
  end
end

@generated function apodization!{T,D}(p::NFFTPlan{D}, f::AbstractArray{T,D}, g::StridedArray{T,D})
	quote
		@nexprs $D d -> offset_d = round(Int, p.n[d] - p.N[d] / 2) - 1

		@nloops $D l f begin
			v = @nref $D f l
			@nexprs $D d -> v *= p.windowHatInvLUT[d][l_d]
			(@nref $D g d -> rem(l_d+offset_d, p.n[d]) + 1) = v
		end
	end
end


### apodization_adjoint! ###

function apodization_adjoint!{T}(p::NFFTPlan{1}, g::AbstractVector{T}, f::StridedVector{T})
  n = p.n[1]
  N = p.N[1]
  const offset = round( Int, n - N / 2 ) - 1
  for l=1:N
    f[l] = g[((l+offset)% n) + 1] * p.windowHatInvLUT[1][l]
  end
end

function apodization_adjoint!{T}(p::NFFTPlan{2}, g::AbstractMatrix{T}, f::StridedMatrix{T})
  n1 = p.n[1]
  N1 = p.N[1]
  n2 = p.n[2]
  N2 = p.N[2]
  const offset1 = round( Int, n1 - N1 / 2 ) - 1
  const offset2 = round( Int, n2 - N2 / 2 ) - 1
  for ly=1:N2
    for lx=1:N1
      f[lx, ly] = g[((lx+offset1)% n1) + 1, ((ly+offset2)% n2) + 1] * p.windowHatInvLUT[1][lx] * p.windowHatInvLUT[2][ly]
    end
  end
end

function apodization_adjoint!{T}(p::NFFTPlan{3}, g::AbstractArray{T,3}, f::StridedArray{T,3})
  n1 = p.n[1]
  N1 = p.N[1]
  n2 = p.n[2]
  N2 = p.N[2]
  n3 = p.n[3]
  N3 = p.N[3]

  const offset1 = round( Int, n1 - N1 / 2 ) - 1
  const offset2 = round( Int, n2 - N2 / 2 ) - 1
  const offset3 = round( Int, n3 - N3 / 2 ) - 1
  for lz=1:N3
    for ly=1:N2
      for lx=1:N1
        f[lx, ly, lz] = g[((lx+offset1)% n1) + 1, ((ly+offset2)% n2) + 1, ((lz+offset3)% n3) + 1] * p.windowHatInvLUT[1][lx] * p.windowHatInvLUT[2][ly] * p.windowHatInvLUT[3][lz]
      end
    end
  end
end

function apodization_adjoint!{T,D}(p::NFFTPlan{D}, g::AbstractArray{T,D}, f::StridedArray{T,D})
  const offset = ntuple(d-> round( Int, p.n[d] - p.N[d] / 2 ) - 1, D)
  idx = Array(Int, D)
  for l=1:prod(p.N)
    it = ind2sub(p.N,l)

    windowHatInvLUTProd = 1.0

    for d=1:D
      idx[d] = ((it[d]+offset[d])% p.n[d]) + 1
      windowHatInvLUTProd *= p.windowHatInvLUT[d][it[d]] 
    end
 
    f[it...] = g[idx...] * windowHatInvLUTProd
  end
end


function sdc{D,T}(p::NFFTPlan{D,T}; iters=20)
  # Weights for sample density compensation.
  # Uses method of Pipe & Menon, 1999. Mag Reson Med, 186, 179.
  weights = ones(Complex{T}, p.M)
  weights_tmp = similar(weights)
  # Pre-weighting to correct non-uniform sample density
  for i in 1:iters
    p.tmpVec[:] = 0.0
    convolve_adjoint!(p, weights, p.tmpVec)
    weights_tmp[:] = 0.0
    convolve!(p, p.tmpVec, weights_tmp)
    for j in 1:length(weights)
      weights[j] = weights[j] / (abs(weights_tmp[j]) + eps(T))
    end
  end
  # Post weights to correct image scaling
  # This finds c, where ||u - c*v||_2^2 = 0 and then uses
  # c to scale all weights by a scalar factor.
  u = ones(Complex{T}, p.N)
  f = nfft(p, u)
  f = f .* weights # apply weights from above
  v = nfft_adjoint(p, f)
  c = v[:] \ u[:]  # least squares diff
  abs(weights * c[1]) # [1] needed b/c 'c' is a 1x1 Array
end

### performance test ###

function nfft_performance()

  m = 4
  sigma = 2.0

  # 1D

  N = 2^19
  M = N

  x = rand(M) .- 0.5
  fHat = rand(M)*1im

  println("NFFT Performance Test 1D")

  tic()
  p = NFFTPlan(x,N,m,sigma)
  println("initialization")
  toc()

  tic()
  fApprox = nfft_adjoint(p,fHat)
  println("adjoint")
  toc()

  tic()
  fHat2 = nfft(p, fApprox);
  println("trafo")
  toc()

  N = 1024
  M = N*N

  x2 = rand(2,M) .- 0.5
  fHat = rand(M)*1im

  println("NFFT Performance Test 2D")

  tic()
  p = NFFTPlan(x2,(N,N),m,sigma)
  println("initialization")
  toc()

  tic()
  fApprox = nfft_adjoint(p,fHat)
  println("adjoint")
  toc()

  tic()
  fHat2 = nfft(p, fApprox);
  println("trafo")
  toc()
  
  N = 32
  M = N*N*N

  x3 = rand(3,M) .- 0.5
  fHat = rand(M)*1im

  println("NFFT Performance Test 3D")

  tic()
  p = NFFTPlan(x3,(N,N,N),m,sigma)
  println("initialization")
  toc()

  tic()
  fApprox = nfft_adjoint(p,fHat)
  println("adjoint")
  toc()

  tic()
  fHat2 = nfft(p, fApprox);
  println("trafo")
  toc()  

end



end
