# NNFFT against the NNDFT: nodes off-centre in both domains, both precisions, adjoint.
@testset "NNFFT" begin
  for (D, J, N, Ys, Ks) in [(1, 300, 200, 10.0, 30.0), (2, 200, 150, 5.0, 20.0), (3, 150, 100, 3.0, 6.0),
                            (1, 50, 70, 0.3, 1.0)], T in (Float32, Float64)
    rng = Xoshiro(1)
    y = (rand(rng, D, N) .- 0.3) .* 2Ys .+ 1.7
    k = (rand(rng, D, J) .- 0.6) .* 2Ks .- 4.2
    f = randn(rng, ComplexF64, N); v = randn(rng, ComplexF64, J)
    pd = NNDFTPlan(k, y)
    for reltol in (T == Float32 ? (1e-3, 1e-5) : (1e-4, 1e-8, 1e-11))
      p = plan_nnfft(T.(k), T.(y); reltol)
      @test size(p) == (J, N)
      @test p * Complex{T}.(f) isa Vector{Complex{T}}
      @test relerr(p * Complex{T}.(f), pd * f) < allowed(accuracy(p), T, (N,))
      @test relerr(adjoint(p) * Complex{T}.(v), adjoint(pd) * v) < allowed(accuracy(p), T, (N,))
      reltol >= 1e-8 && @test accuracy(p) < 2reltol
    end
  end
  # all targets equal, a single source
  p = plan_nnfft(fill(0.3, 1, 5), fill(-2.0, 1, 1))
  @test p * [1.0 + 0im] ≈ fill(cispi(-2 * 0.3 * -2.0), 5)
  @test_throws ArgumentError plan_nnfft(rand(2, 4), rand(1, 4))
  @test_throws ArgumentError plan_nnfft(rand(2, 4), rand(2, 4); σ=0.9)
  # a copy computes the same; new targets are planned from the sources in full precision
  rng = Xoshiro(2)
  y = 50 .* randn(rng, 2, 300); k = 3 .* randn(rng, 2, 200)
  p = plan_nnfft(k, y; eltype=Float32)
  f = randn(rng, ComplexF32, 300)
  @test copy(p) * f == p * f
  q = copy(p)
  k2 = 3 .* randn(rng, 2, 100)
  nodes!(q, k2)
  @test relerr(q * f, NNDFTPlan(k2, y) * ComplexF64.(f)) < 10 * accuracy(q)
  @test p * f == copy(p) * f
  @test_throws DimensionMismatch nodes!(q, rand(3, 10))

  @testset "batch dimensions" begin
    rng = Xoshiro(3)
    y = 20 .* randn(rng, 2, 120); k = 2 .* randn(rng, 2, 90)
    q = plan_nnfft(k, y; reltol=1e-9)
    for (N, dims) in (((120, 3), 1), ((4, 120), 2), ((2, 120, 3), 2))
      p = plan_nnfft(k, y, N; dims, reltol=1e-9)
      NOut = Base.setindex(N, 90, dims)
      @test size_in(p) == N && size_out(p) == NOut
      f = randn(rng, ComplexF64, N); v = randn(rng, ComplexF64, NOut)
      g = p * f; h = adjoint(p) * v
      for I in CartesianIndices(Base.setindex(N, 1, dims))
        sl = ntuple(d -> d == dims ? Colon() : I[d], length(N))
        @test g[sl...] ≈ q * f[sl...]
        @test h[sl...] ≈ adjoint(q) * v[sl...]
      end
      @test copy(p) * f == g
      c = copy(p); nodes!(c, k[:, 1:50])
      @test size_out(c) == Base.setindex(N, 50, dims)
    end
    @test_throws ArgumentError plan_nnfft(k, y, (120, 3))                 # dims needed
    @test_throws DimensionMismatch plan_nnfft(k, y, (100, 3); dims=1)     # wrong number of sources
    @test_throws ArgumentError plan_nnfft(k, y, (120, 3); dims=3)
  end
end
