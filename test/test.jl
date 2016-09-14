using Base.Test
using NFFT

eps = 1e-5
m = 4
sigma = 2.0

for N in [(128,), (16,16), (12,12,12), (6,6,6,6)]
  D = length(N)
  println("Testing in ", D, " dimensions...")

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


# NFFT along a specified dimension should give the same result as
# running a 1D NFFT on every slice along that dimension
for D in 2:3
	println("Testing directional NFFT in ", D, " dimensions...")

	N = tuple( 2*rand(4:8,D)... )
	M = prod(N)
	for d in 1:D
		x = rand(M) - 0.5

		f = rand(N) + rand(N)*im
		p_dir = NFFTPlan(x, d, N)
		fHat_dir = nfft(p_dir, f)
		g_dir = nfft_adjoint(p_dir, fHat_dir)

		p = NFFTPlan(x, N[d])
		fHat = similar(fHat_dir)
		g = similar(g_dir)

		sz = size(fHat)
		Rpre = CartesianRange( sz[1:d-1] )
		Rpost = CartesianRange( sz[d+1:end] )
		for Ipost in Rpost, Ipre in Rpre
			idx = [Ipre, :, Ipost]
			fview = f[idx...]
			fHat[idx...] = nfft(p, vec(fview))

			fHat_view = fHat_dir[idx...]
			g[idx...] = nfft_adjoint(p, vec(fHat_view))
		end

		e = norm( fHat_dir[:] - fHat[:] )
		@test_approx_eq e 0

		e = norm( g_dir[:] - g[:] ) / norm(g[:])
		@test e < eps
	end
end

