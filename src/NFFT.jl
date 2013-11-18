module NFFT

import Base.ind2sub

export NFFTPlan, nfft, nfft_adjoint, ndft, ndft_adjoint, nfft_test, nfft_performance



function window_kaiser_bessel(x,n,m,sigma)
  b = pi*(2-1/sigma)
  arg = m^2-n^2*x^2
  if(abs(x) < m/n)
    y=sinh(b*sqrt(arg))/sqrt(arg)/pi
  elseif(abs(x) > m/n)
    y=0
  else
    y=b/pi
  end
  return y
end


function window_kaiser_bessel_hat(k,n,m,sigma)
  b=pi*(2-1/sigma);
  return besseli(0,m*sqrt(b^2-(2*pi*k/n)^2));
end

type NFFTPlan{T,Dim}
  D::Int64
  T::Type
  N::NTuple{Dim,Int64}
  M::Int64
  x::Array{T,2}
  m::Int64
  sigma::T
  n::NTuple{Dim,Int64}
  K::Int64
  windowLUT::Vector{Vector{T}}
  windowHatInvLUT::Vector{Vector{T}}
  tmpVec::Array{Complex{T},Dim}
end

function NFFTPlan{T,D}(x::Array{T,2}, N::NTuple{D,Int64}, m=4, sigma=2.0)
  
  if D != size(x,1)
    throw(ArgumentError())
  end

  n = ntuple(D, d->int(round(sigma*N[d])) )

  tmpVec = zeros(Complex{T}, n)

  M = size(x,2)

  # Create lookup table
  K = 1000
  
  windowLUT = Array(Vector{T},D)
  for d=1:D
    Z = int(3*K/2)
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


  NFFTPlan(D, T, N, M, x, m, sigma, n, K, windowLUT, windowHatInvLUT, tmpVec )
end

function NFFTPlan{T}(x::Array{T,1}, N::Integer, m=4, sigma=2.0)
  NFFTPlan(reshape(x,1,length(x)), (N,), m, sigma)
end

### nfft functions ###

function nfft!{T,D}(p::NFFTPlan, f::Array{T,D}, fHat::Vector{T})
  p.tmpVec[:] = 0
  @inbounds apodization!(p, f, p.tmpVec)
  fft!(p.tmpVec)
  @inbounds convolve!(p, p.tmpVec, fHat)
  return fHat
end

function nfft{T,D}(p::NFFTPlan, f::Array{T,D})
  fHat = zeros(T, p.M)
  nfft!(p, f, fHat)
  return fHat
end

function nfft{T,D}(x, f::Array{T,D})
  p = NFFTPlan(x, size(f) )
  return nfft(p, f)
end

function nfft_adjoint!{T,D}(p::NFFTPlan, fHat::Vector{T}, f::Array{T,D})
  p.tmpVec[:] = 0
tic()
  @inbounds convolve_adjoint!(p, fHat, p.tmpVec)
toc()
tic()
  ifft!(p.tmpVec)
  p.tmpVec *= prod(p.n)
toc()
tic()
  @inbounds apodization_adjoint!(p, p.tmpVec, f)
toc()
  return f
end

function nfft_adjoint{T,D}(p::NFFTPlan{T,D}, fHat::Vector{Complex{T}})
  f = zeros(Complex{T},p.N)
  nfft_adjoint!(p, fHat, f)
  return f
end

function nfft_adjoint{T,D}(x, N::NTuple{D,Int64}, fHat::Vector{T})
  p = NFFTPlan(x, N)
  return nfft_adjoint(p, fHat)
end

### ndft functions ###

function ind2sub{T}(::Array{T,1}, idx)
  idx
end

function ndft{T,D}(plan::NFFTPlan, f::Array{T,D})
  g = zeros(T, plan.M)

  for l=1:prod(plan.N)
    idx = ind2sub(plan.N,l)

    for k=1:plan.M
      arg = zero(T)
      for d=1:plan.D
        arg += plan.x[d,k] * ( idx[d] - 1 - plan.N[d] / 2 )
      end
      g[k] += f[l] * exp(-2*pi*1im*arg)
    end
  end

  return g
end

function ndft_adjoint{T}(plan::NFFTPlan, fHat::Array{T,1})

  g = zeros(Complex{plan.T}, plan.N)

  for l=1:prod(plan.N)
    idx = ind2sub(plan.N,l)

    for k=1:plan.M
      arg = zero(T)
      for d=1:plan.D
        arg += plan.x[d,k] * ( idx[d] - 1 - plan.N[d] / 2 )
      end
      g[l] += fHat[k] * exp(2*pi*1im*arg)
    end
  end

  return g
end



### convolve! ###

function convolve!{T}(p::NFFTPlan, g::Array{T,1}, fHat::Array{T,1})
  n = p.n[1]

  for k=1:p.M # loop over nonequispaced nodes
    c = int(floor(p.x[k]*n))
    for l=(c-p.m):(c+p.m) # loop over nonzero elements

      idx = ((l+n)% n) + 1
      idx2 = abs(((p.x[k]*n - l)/p.m )*(p.K-1)) + 1
      idx2L = int(floor(idx2))

      fHat[k] += g[idx] * (p.windowLUT[1][idx2L] + ( idx2-idx2L ) * (p.windowLUT[1][idx2L+1] - p.windowLUT[1][idx2L] ) )
    end
  end
end

function convolve!{T}(p::NFFTPlan, g::Array{T,2}, fHat::Array{T,1})
  scale = 1.0 / p.m * (p.K-1)

  n1 = p.n[1]
  n2 = p.n[2]

  gPtr = pointer(g) 

  for k=1:p.M # loop over nonequispaced nodes
    c0 = int(floor(p.x[1,k]*n1))
    c1 = int(floor(p.x[2,k]*n2))

    for l1=(c1-p.m):(c1+p.m) # loop over nonzero elements

      idx1 = ((l1+n2)% n2) + 1

      idx2 = abs((p.x[2,k]*n2 - l1)*scale) + 1
      idx2L = int(floor(idx2))

      tmpWin = (p.windowLUT[2][idx2L] + ( idx2-idx2L ) * (p.windowLUT[2][idx2L+1] - p.windowLUT[2][idx2L] ) )

      tt=(idx1-1)*n1

      for l0=(c0-p.m):(c0+p.m)

        idx0 = ((l0+n1)% n1) + 1
        idx2 = abs((p.x[1,k]*n1 - l0)*scale) + 1
        idx2L = int(idx2)

        tmp = unsafe_load(gPtr, idx0+tt)
        #tmp = g[idx0,idx1]
        fHat[k] += tmp * tmpWin * (p.windowLUT[1][idx2L] + ( idx2-idx2L ) * (p.windowLUT[1][idx2L+1] - p.windowLUT[1][idx2L] ) )
      end
    end
  end
end

function convolve!{T,D}(p::NFFTPlan, g::Array{T,D}, fHat::Array{T,1})
  l = Array(Int64,p.D)
  idx = Array(Int64,p.D)
  P = Array(Int64,p.D)
  c = Array(p.T,p.D)

  for k=1:p.M # loop over nonequispaced nodes

    for d=1:D
      c[d] = int(floor(p.x[d,k]*p.n[d]))
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
        idx2L = int(floor(idx2))
        tmp *= (p.windowLUT[d][idx2L] + ( idx2-idx2L ) * (p.windowLUT[d][idx2L+1] - p.windowLUT[d][idx2L] ) )
      end

      fHat[k] += tmp;
    end
  end
end


### convolve_adjoint! ###

function convolve_adjoint!{T}(p::NFFTPlan, fHat::Array{T,1}, g::Array{T,1})
  n = p.n[1]

  println(p.m)

  for k=1:p.M # loop over nonequispaced nodes
    c = int(floor(p.x[k]*n))
    for l=(c-p.m):(c+p.m) # loop over nonzero elements

      idx = ((l+n)%n)+1
      idx2 = 1#abs(((p.x[k]*n - l)/p.m )*(p.K-1)) + 1
      idx2L = int(idx2)

      g[idx] += fHat[k] * (p.windowLUT[1][idx2L] + ( idx2-idx2L ) * (p.windowLUT[1][idx2L+1] - p.windowLUT[1][idx2L] ) )
    end
  end
end

function convolve_adjoint!{T}(p::NFFTPlan, fHat::Array{T,1}, g::Array{T,2})
  scale = 1.0 / p.m * (p.K-1)
  n1 = p.n[1]
  n2 = p.n[2]

  gPtr = pointer(g) 

  for k=1:p.M # loop over nonequispaced nodes
    c0 = int(floor(p.x[1,k]*n1))
    c1 = int(floor(p.x[2,k]*n2))

    for l1=(c1-p.m):(c1+p.m) # loop over nonzero elements
      idx1 = ((l1+n2)%n2) + 1
      idx2 = abs((p.x[2,k]*n2 - l1)*scale) + 1
      idx2L = int(floor(idx2))

      tmp = fHat[k] * (p.windowLUT[2][idx2L] + ( idx2-idx2L ) * (p.windowLUT[2][idx2L+1] - p.windowLUT[2][idx2L] ) )

      tt=(idx1-1)*n1

      for l0=(c0-p.m):(c0+p.m)
        idx0 = ((l0+n1)%n1) + 1
        idx2 = abs((p.x[1,k]*n1 - l0)*scale) + 1
        idx2L = int(idx2)
        #g[idx0,idx1] += tmp * (p.windowLUT[1][idx2L] + ( idx2-idx2L ) * (p.windowLUT[1][idx2L+1] - p.windowLUT[1][idx2L] ) )
        tmp *= (p.windowLUT[1][idx2L] + ( idx2-idx2L ) * (p.windowLUT[1][idx2L+1] - p.windowLUT[1][idx2L] ) )
        tmpG = unsafe_load(gPtr, idx0+tt) + tmp
        unsafe_store!(gPtr, tmpG, idx0+tt)
      end
    end
  end
end

function convolve_adjoint!{T,D}(p::NFFTPlan, fHat::Array{T,1}, g::Array{T,D})
  l = Array(Int64,p.D)
  idx = Array(Int64,p.D)
  P = Array(Int64,p.D)
  c = Array(p.T,p.D)

  for k=1:p.M # loop over nonequispaced nodes

    for d=1:D
      c[d] = int(floor(p.x[d,k]*p.n[d]))
      P[d] = 2*p.m + 1
    end

    for j=1:prod(P) # loop over nonzero elements
      it = ind2sub(tuple(P...),j)
      for d=1:D
        l[d] = c[d]-p.m+it[d]
        idx[d] = ((l[d]+p.n[d])%p.n[d]) 
      end

      tmp = fHat[k]
      for d=1:D
        idx2 = abs(((p.x[d,k]*p.n[d] - l[d])/p.m )*(p.K-1)) + 1
        idx2L = int(floor(idx2))
        tmp *= (p.windowLUT[d][idx2L] + ( idx2-idx2L ) * (p.windowLUT[d][idx2L+1] - p.windowLUT[d][idx2L] ) )
      end

      g[idx...] += tmp;
    end
  end
end





### apodization! ###

function apodization!{T}(p::NFFTPlan, f::Array{T,1}, g::Array{T,1})
  n = p.n[1]
  N = p.N[1]
  const offset = int( n - N / 2 ) - 1
  for l=1:N
    g[((l+offset)% n) + 1] = f[l] * p.windowHatInvLUT[1][l]
  end
end

function apodization!{T}(p::NFFTPlan, f::Array{T,2}, g::Array{T,2})
  n1 = p.n[1]
  N1 = p.N[1]
  n2 = p.n[2]
  N2 = p.N[2]
  const offset1 = int( n1 - N1 / 2 ) - 1
  const offset2 = int( n2 - N2 / 2 ) - 1
  for ly=1:N2
    for lx=1:N1
      g[((lx+offset1)% n1) + 1, ((ly+offset2)% n2) + 1] = f[lx, ly]  *   p.windowHatInvLUT[1][lx] * p.windowHatInvLUT[2][ly]
    end
  end
end

function apodization!{T,D}(p::NFFTPlan, f::Array{T,D}, g::Array{T,D})
  const offset = ntuple(D, d-> int( p.n[d] - p.N[d] / 2 ) - 1)
  idx = similar(p.N)
  for l=1:prod(p.N)
    it = ind2sub(p.N,l)

    windowHatInvLUTProd = 1.0

    for d=1:p.D
      idx[d] = ((it[d]+offset[d])% p.n[d]) + 1
      windowHatInvLUTProd *= p.windowHatInvLUT[d][it[d]] 
    end
 
    g[idx...] = f[it...] * windowHatInvLUTProd
  end
end


### apodization_adjoint! ###

function apodization_adjoint!{T}(p::NFFTPlan, g::Array{T,1}, f::Array{T,1})
  n = p.n[1]
  N = p.N[1]
  const offset = int( n - N / 2 ) - 1
  for l=1:N
    f[l] = g[((l+offset)% n) + 1] = p.windowHatInvLUT[1][l]
  end
end

function apodization_adjoint!{T}(p::NFFTPlan, g::Array{T,2}, f::Array{T,2})
  n1 = p.n[1]
  N1 = p.N[1]
  n2 = p.n[2]
  N2 = p.N[2]
  const offset1 = int( n1 - N1 / 2 ) - 1
  const offset2 = int( n2 - N2 / 2 ) - 1
  for ly=1:N2
    for lx=1:N1
      f[lx, ly] = g[((lx+offset1)% n1) + 1, ((ly+offset2)% n2) + 1] * p.windowHatInvLUT[1][lx] * p.windowHatInvLUT[2][ly]
    end
  end
end

function apodization_adjoint!{T,D}(p::NFFTPlan, g::Array{T,D}, f::Array{T,D})
  const offset = ntuple(D, d-> int( p.n[d] - p.N[d] / 2 ) - 1)
  idx = similar(p.N)
  for l=1:prod(p.N)
    it = ind2sub(p.N,l)

    windowHatInvLUTProd = 1.0

    for d=1:p.D
      idx[d] = ((it[d]+offset[d])% p.n[d]) + 1
      windowHatInvLUTProd *= p.windowHatInvLUT[d][it[d]] 
    end
 
    f[it...] = g[idx...] * windowHatInvLUTProd
  end
end

### test functions ###


function nfft_test()

  m = 6
  sigma = 2.0

  # 1D

  N = 16
  x = linspace(-0.4, 0.4, N)
  fHat = linspace(0,1,N)*1im
  p = NFFTPlan(x, N, m, sigma);

  f = ndft_adjoint(p, fHat)
  fApprox = nfft_adjoint(p, fHat)
  println( norm(f-fApprox) / norm(f) )

  gHat = ndft(p, f)
  gHatApprox = nfft(p, f)
  println( norm(gHat-gHatApprox) / norm(gHat) )

  # 2D

  N = (4,4)
  M = 16
  x = reshape(linspace(-0.4, 0.4, 2*M), 2, M)
  fHat = linspace(0,1,M)*1im
  p = NFFTPlan(x, N, m, sigma)

  f = ndft_adjoint(p, fHat)
  fApprox = nfft_adjoint(p, fHat)
  println( norm(f[:]-fApprox[:]) / norm(f[:]) )

  gHat = ndft(p, f)
  gHatApprox = nfft(p, f)
  println( norm(gHat[:]-gHatApprox[:]) / norm(gHat[:]) )

  # 3D

  N = (4,4,4)
  M = 4^3
  x = reshape(linspace(-0.4, 0.4, 3*M), 3, M)
  fHat = linspace(0,1,M)*1im
  p = NFFTPlan(x, N, m, sigma)

  f = ndft_adjoint(p, fHat)
  fApprox = nfft_adjoint(p, fHat)
  println( norm(f[:]-fApprox[:]) / norm(f[:]) )

  gHat = ndft(p, f)
  gHatApprox = nfft(p, f)
  println( norm(gHat[:]-gHatApprox[:]) / norm(gHat[:]) )


end

function nfft_performance()

  m = 4
  sigma = 2.0

  # 1D

  N = 2^19
  M = N

  x = rand(M) - 0.5
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

  x2 = rand(2,M) - 0.5
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

end



end
