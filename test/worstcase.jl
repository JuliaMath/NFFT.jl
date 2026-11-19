# The accuracy contract in its worst case: all nodes at the same fractional offset t of a cell of
# the oversampled grid, so that the aliases of a mode add up coherently at every node, and a single
# mode as input (the band edge, the centre, a quarter). The relative error is then the error per
# node, which the bound (1 + est)^D − 1 of the planner must cover.

@testset "Worst case: single modes, nodes at one grid offset" begin
  rng = Xoshiro(5)
  for (N, tols) in (((64,), (1e-4, 1e-8, 1e-12)), ((16, 12), (1e-4, 1e-8, 1e-12)), ((8, 6, 8), (1e-4, 1e-8, 1e-11)))
    D = length(N); J = 40
    for tol in tols
      Ñ = oversampled_size(plan_nfft(rand(rng, D, J) .- 0.5, N; reltol=tol))
      for t in (0.0, 0.25, 0.5, 0.37)
        k = reduce(hcat, [[(rand(rng, 0:Ñ[d]-1) + t) / Ñ[d] - 0.5 for d in 1:D] for _ in 1:J])
        p = plan_nfft(k, N; reltol=tol)
        @test oversampled_size(p) == Ñ
        q = NDFTPlan(k, N)
        for m in (ntuple(_ -> 1, D), map(n -> n ÷ 2 + 1, N), map(n -> n ÷ 4 + 1, N), N)
          f = zeros(ComplexF64, N); f[m...] = 1
          @test relerr(p * f, q * f) <= tol
        end
        y = zeros(ComplexF64, J); y[1] = 1
        @test relerr(adjoint(p) * y, adjoint(q) * y) <= tol
      end
    end
  end

  # all modes and a fine set of offsets for a case whose kernel the alias sum est_wc underestimated
  # (8 × 8 at 1e-6: 1.11·tol with est_wc as the bound)
  N, tol, J = (8, 8), 1e-6, 48
  Ñ = oversampled_size(plan_nfft(rand(rng, 2, J) .- 0.5, N; reltol=tol))
  for t in 0:0.05:0.95
    k = reduce(hcat, [[(mod(7j * (d + 2), Ñ[d]) + t) / Ñ[d] - 0.5 for d in 1:2] for j in 1:J])
    p, q = plan_nfft(k, N; reltol=tol), NDFTPlan(k, N)
    worst = maximum(CartesianIndices(N)) do I
      f = zeros(ComplexF64, N); f[I] = 1
      relerr(p * f, q * f)
    end
    @test worst <= tol
  end
end

