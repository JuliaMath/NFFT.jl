@testset "NFCT and NFST" begin
  rng = Xoshiro(7)

  @testset "$(nameof(P)) $N $T" for (P, Pd) in ((NFCTPlan, NDCTPlan), (NFSTPlan, NDSTPlan)),
                                      N in ((1,), (2,), (64,), (33,), (16, 20), (8, 9, 10)),
                                      T in (Float64, Float32)
    D = length(N); J = 150
    k = T.(rand(rng, D, J) ./ 2)
    k[:, 1] .= 0; k[:, 2] .= 0.5                     # end points of the node range
    reltol = T == Float64 ? 1e-9 : 1e-5
    p = P(k, N; reltol)
    pd = Pd(Float64.(k), N)
    @test p isa P{T,D,1}
    @test size_in(p) == N && size_out(p) == (J,)
    @test size(p) == (J, prod(N))
    f = randn(rng, T, N); fHat = randn(rng, T, J)
    y = p * f
    @test eltype(y) == T
    @test relerr(y, pd * Float64.(f)) < allowed(accuracy(p), T, N)
    g = transpose(p) * fHat
    @test size(g) == N
    # relative to the typical size of the result: for N = (1,) the single sum can cancel
    gd = transpose(pd) * Float64.(fHat)
    @test norm(g - gd) / max(norm(gd), norm(fHat) * sqrt(prod(N)) / 2) < allowed(accuracy(p), T, N)
    @test abs(dot(fHat, y) - dot(g, f)) < 100eps(T) * norm(fHat) * norm(y)
  end

  @testset "accuracy over reltol" begin
    k = rand(rng, 2, 500) ./ 2
    f = randn(rng, 24, 30)
    for reltol in (1e-3, 1e-6, 1e-12)
      p = plan_nfct(k, size(f); reltol)
      @test relerr(p * f, NDCTPlan(k, size(f)) * f) < 10reltol
      p = plan_nfst(k, size(f); reltol)
      @test relerr(p * f, NDSTPlan(k, size(f)) * f) < 10reltol
    end
  end

  @testset "options, copy, nodes!" begin
    k = rand(rng, 1, 300) ./ 2
    f = randn(rng, 100)
    p = plan_nfct(k, 100; W = 8, σ = 2.0)
    @test NFFT.parameters(p).W == 8 && NFFT.parameters(p).σ == 2.0
    @test relerr(p * f, NDCTPlan(k, (100,)) * f) < 1e-5
    q = copy(p)
    @test q * f == p * f
    k2 = rand(rng, 1, 200) ./ 2
    nodes!(q, k2)
    @test size(q) == (200, 100)
    @test relerr(q * f, NDCTPlan(k2, (100,)) * f) < 1e-5
    @test nodes(q) === k2
    @test_throws ArgumentError plan_nfct(k .- 0.25, 100)
    @test_throws ArgumentError plan_nfst(k, 100; m = 4)
    @test_throws DimensionMismatch p * randn(rng, 99)
    @test nfct(k, f) ≈ p * f rtol = 1e-6
    @test nfst_transpose(k, 100, randn(rng, 300)) isa Vector{Float64}
  end

  @testset "dims" begin
    k = rand(rng, 1, 120) ./ 2
    f = randn(rng, 3, 40, 5)
    for (P, Pd) in ((NFCTPlan, NDCTPlan), (NFSTPlan, NDSTPlan))
      p = P(k, size(f); dims = 2)
      pd = Pd(k, (40,))
      y = p * f
      @test size(y) == (3, 120, 5)
      @test relerr(y[2, :, 4], pd * f[2, :, 4]) < 1e-7
      g = transpose(p) * y
      @test relerr(g[3, :, 1], transpose(pd) * y[3, :, 1]) < 1e-7
    end
  end
end
