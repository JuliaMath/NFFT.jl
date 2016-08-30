using Base.Test
using NFFT

eps = 1e-5
m = 4
sigma = 2.0

D = 0
for N in [(128,), (16,16), (12,12,12), (6,6,6,6)]
  D += 1
  @printf("Testing in dimension %u...\n", D)

  M = prod(N)
  x = reshape(linspace(-0.4, 0.4, D*M), D, M)
  if !(typeof(x) <: Array)
	  x = collect(x)
  end
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

