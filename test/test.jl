using Base.Test
using NFFT

eps = 1e-5
m = 6
sigma = 2.0

# 1D
begin
  N = 16
  x = linspace(-0.4, 0.4, N)
  fHat = linspace(0,1,N)*1im
  p = NFFTPlan(x, N, m, sigma);

  f = ndft_adjoint(p, fHat)
  fApprox = nfft_adjoint(p, fHat)
  @test norm(f-fApprox) / norm(f) < eps

  gHat = ndft(p, f)
  gHatApprox = nfft(p,f)

  @test norm(gHat-gHatApprox) / norm(gHat)  < eps
end

# 2D
begin
  N = (4,4)
  M = 16
  x = reshape(linspace(-0.4, 0.4, 2*M), 2, M)
  fHat = linspace(0,1,M)*1im
  p = NFFTPlan(x, N, m, sigma)

  f = ndft_adjoint(p, fHat)
  fApprox = nfft_adjoint(p, fHat)
  @test norm(f[:]-fApprox[:]) / norm(f[:]) < eps

  gHat = ndft(p, f)
  gHatApprox = nfft(p, f)
  @test norm(gHat[:]-gHatApprox[:]) / norm(gHat[:]) < eps
end

# 3D
begin
  N = (8,8,8)
  M = 8^3
  x = reshape(linspace(-0.4, 0.4, 3*M), 3, M)
  fHat = linspace(0,1,M)*1im
  p = NFFTPlan(x, N, m, sigma)

  f = ndft_adjoint(p, fHat)
  fApprox = nfft_adjoint(p, fHat)
  @test norm(f[:]-fApprox[:]) / norm(f[:]) < eps

  gHat = ndft(p, f)
  gHatApprox = nfft(p, f)
  @test norm(gHat[:]-gHatApprox[:]) / norm(gHat[:]) < eps
end


