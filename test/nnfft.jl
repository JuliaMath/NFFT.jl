# NNFFT against the NNDFT: nodes off-centre in both domains, both precisions, adjoint.
@testset "NNFFT" begin
  relerr(a, b) = norm(a - b) / norm(b)
  for (D, J, N, Ys, Ks) in [(1, 300, 200, 10.0, 30.0), (2, 200, 150, 5.0, 20.0), (3, 150, 100, 3.0, 6.0),
                            (1, 50, 70, 0.3, 1.0)], T in (Float32, Float64)
    rng = MersenneTwister(1)
    y = (rand(rng, D, N) .- 0.3) .* 2Ys .+ 1.7
    k = (rand(rng, D, J) .- 0.6) .* 2Ks .- 4.2
    f = randn(rng, ComplexF64, N); v = randn(rng, ComplexF64, J)
    pd = NNDFTPlan(k, y)
    for reltol in (T == Float32 ? (1e-3, 1e-5) : (1e-4, 1e-8, 1e-12))
      p = plan_nnfft(T.(k), T.(y); reltol)
      @test size(p) == (J, N)
      @test p * Complex{T}.(f) isa Vector{Complex{T}}
      @test relerr(p * Complex{T}.(f), pd * f) < 10 * accuracy(p)
      @test relerr(adjoint(p) * Complex{T}.(v), adjoint(pd) * v) < 10 * accuracy(p)
      reltol >= 1e-8 && @test accuracy(p) < 2reltol
    end
  end
  # all targets equal, a single source
  p = plan_nnfft(fill(0.3, 1, 5), fill(-2.0, 1, 1))
  @test p * [1.0 + 0im] ≈ fill(cispi(-2 * 0.3 * -2.0), 5)
  @test_throws ArgumentError plan_nnfft(rand(2, 4), rand(1, 4))
end
