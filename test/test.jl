using Base.Test
using NFFT

eps = 1e-5
m = 4
sigma = 2.0

# 1D
begin
  N = 128
  x = collect( linspace(-0.4, 0.4, N) )
  fHat = collect( linspace(0,1,N)*1im )
  p = NFFTPlan(x, N, m, sigma);

  f = ndft_adjoint(p, fHat)
  fApprox = nfft_adjoint(p, fHat)
  e = norm(f-fApprox) / norm(f)
  println(e)
  @test e < eps

  gHat = ndft(p, f)
  gHatApprox = nfft(p,f)

  e = norm(gHat-gHatApprox) / norm(gHat) 
  println(e)
  @test e < eps
end

# 2D
begin
  N = (16,16)
  M = prod(N)
  x = collect(reshape(linspace(-0.4, 0.4, 2*M), 2, M))
  fHat = linspace(0,1,M)*1im
  p = NFFTPlan(x, N, m, sigma)

  f = ndft_adjoint(p, fHat)
  fApprox = nfft_adjoint(p, fHat)
  e = norm(f[:]-fApprox[:]) / norm(f[:])
  println(e)
  @test e < eps

  gHat = ndft(p, f)
  gHatApprox = nfft(p, f)
  e = norm(gHat[:]-gHatApprox[:]) / norm(gHat[:])
  println(e)
  @test e < eps
end

# 3D
begin
  N = (12,12,12)
  M = prod(N)
  x = collect(reshape(linspace(-0.4, 0.4, 3*M), 3, M))
  fHat = linspace(0,1,M)*1im
  p = NFFTPlan(x, N, m, sigma)

  f = ndft_adjoint(p, fHat)
  fApprox = nfft_adjoint(p, fHat)
  e = norm(f[:]-fApprox[:]) / norm(f[:])
  println(e)
  @test e < eps

  gHat = ndft(p, f)
  gHatApprox = nfft(p, f)
  e = norm(gHat[:]-gHatApprox[:]) / norm(gHat[:])
  println(e)
  @test e < eps
end


