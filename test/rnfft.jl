@testset "RNFFT" begin
  rng = MersenneTwister(11)
  relerr(a, b) = norm(a - b) / norm(b)
  # f̂_j = Re Σ_{n₁=0}^{N₁÷2} Σ_{n'} h_n exp(−2πi n·k_j), the other dimensions centred
  modes(I, N) = ntuple(d -> d == 1 ? I[1] - 1 : I[d] - 1 - N[d] ÷ 2, length(N))
  rdirect(k, N, h) = [sum(real(h[I] * cispi(-2 * sum(modes(I, N) .* Tuple(k[:, j])))) for I in CartesianIndices(h))
                      for j in 1:size(k, 2)]
  radjoint(k, N, y) = [sum(y[j] * cispi(2 * sum(modes(I, N) .* Tuple(k[:, j]))) for j in eachindex(y))
                       for I in CartesianIndices((N[1] ÷ 2 + 1, Base.tail(N)...))]

  @testset "$N $T" for N in ((1,), (2,), (64,), (33,), (16, 20), (15, 8), (8, 9, 10)), T in (Float64, Float32)
    D = length(N); J = 150
    k = rand(rng, D, J) .- 0.5
    k[:, 1] .= -0.5
    p = plan_rnfft(T.(k), N; reltol = T == Float64 ? 1e-9 : 1e-5)
    @test p isa RNFFTPlan{T,D,1}
    @test size_in(p) == (N[1] ÷ 2 + 1, Base.tail(N)...) && size_out(p) == (J,)
    h = randn(rng, Complex{T}, size_in(p)); y = randn(rng, T, J)
    fHat = p * h
    @test fHat isa Vector{T}
    tol = max(10accuracy(p), 100eps(T))
    @test relerr(fHat, rdirect(Float64.(T.(k)), N, h)) < tol
    g = adjoint(p) * y
    @test g isa Array{Complex{T},D}
    @test relerr(g, radjoint(Float64.(T.(k)), N, y)) < tol
    # adjoint with respect to the real inner product
    @test abs(dot(y, fHat) - real(dot(g, h))) < 100eps(T) * norm(y) * norm(fHat)
  end

  @testset "dims, slices, copy, nodes!" begin
    k = rand(rng, 1, 90) .- 0.5
    q = plan_rnfft(k, 40)
    for (N, nthreads) in (((40, 3), 8), ((40, 64), 8), ((40, 64), 1))
      p = plan_rnfft(k, N; dims = 1, nthreads)
      @test isempty(p.slicecores) == (N[2] < 8 || nthreads == 1)
      h = randn(rng, ComplexF64, size_in(p))
      y = p * h
      @test size(y) == (90, N[2])
      @test all(y[:, l] ≈ q * h[:, l] for l in 1:N[2])
      g = adjoint(p) * y
      @test all(g[:, l] ≈ adjoint(q) * y[:, l] for l in 1:N[2])
      c = copy(p)
      @test c * h == y
      k2 = rand(rng, 1, 50) .- 0.5
      nodes!(c, k2)
      @test size(c * h) == (50, N[2])
      @test (c * h)[:, 2] ≈ plan_rnfft(k2, 40) * h[:, 2]
    end
    p = plan_rnfft(rand(rng, 2, 50) .- 0.5, (6, 7, 4); dims = 2:3)
    @test size_in(p) == (6, 4, 4)
    @test_throws DimensionMismatch p * randn(rng, ComplexF64, 6, 7, 4)
    @test_throws ArgumentError plan_rnfft(k .+ 1, 40)
  end

  @testset "NFCT with slices" begin
    k = rand(rng, 1, 70) ./ 2
    f = randn(rng, 30, 40)
    p = plan_nfct(k, size(f); dims = 1, nthreads = 8)
    @test !isempty(p.slicecores)
    q = plan_nfct(k, 30)
    @test p * f ≈ reduce(hcat, [q * f[:, l] for l in 1:40])
    y = p * f
    @test transpose(p) * y ≈ reduce(hcat, [transpose(q) * y[:, l] for l in 1:40])
  end
end
