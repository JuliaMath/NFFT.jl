using Base.Test
using NFFT

eps = 1e-5
m = 4
sigma = 2.0

for N in [(128,), (16,16), (12,12,12), (6,6,6,6)]
  D = length(N)
  @printf("Testing in %u dimensions...\n", D)

  M = prod(N)
  x = rand(D,M) - 0.5
  p = NFFTPlan(x, N, m, sigma)

  fHat = linspace(0,1,M)*im
  f = ndft_adjoint(p, fHat)
  fApprox = nfft_adjoint(p, fHat)
  e = norm(f[:] - fApprox[:]) / norm(f[:])
  println(e)
  @test e < eps

  gHat = ndft(p, f)
  gHatApprox = nfft(p, f)
  e = norm(gHat[:] - gHatApprox[:]) / norm(gHat[:])
  println(e)
  @test e < eps
end

