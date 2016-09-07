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


# NFFT along a specified dimension
# In an array with repeated entries the result should be the same as
# repeating the output of 1D NFFT
for N in [(16,16), (8,8,8)]
	M = prod(N)
	D = length(N)
	for d in 1:D
		x = rand(M) - 0.5

		p = NFFTPlan(x, N[d])
		f = rand(N[d]) + rand(N[d])*im
		fHat = nfft(p, f)

		reshape_dim = ones(Int,D)
		reshape_dim[d] = N[d]
		f = reshape(f, (reshape_dim...))
		rep_dim = [N...]
		rep_dim[d] = 1
		f_rep = repeat(f, outer=rep_dim)
		p_dir = NFFTDirPlan(x, N, d)
		fHat_dir = nfft(p_dir, f_rep)

		reshape_dim[d] = M
		fHat = reshape(fHat, (reshape_dim...))
		e = norm( repeat(fHat, outer=rep_dim)[:] - fHat_dir[:] )
		@test_approx_eq e 0
	end
end

