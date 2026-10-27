# The stage interface (Knopp et al. 2023): resampling correction → FFT → resampling and the adjoint
# steps compose to the transform.

@testset "Stages" begin
  rng = Xoshiro(11)
  for N in ((40,), (17, 22), (9, 10, 8))
    D = length(N); J = 150
    k = rand(rng, D, J) .- 0.5
    p = plan_nfft(k, N; reltol=1e-9)
    Ñ = oversampled_size(p)
    @test length(Ñ) == D && all(Ñ .> N)
    f = randn(rng, ComplexF64, N); fHat = randn(rng, ComplexF64, J)

    g = zeros(ComplexF64, Ñ); out = zeros(ComplexF64, J)
    resampling_correction!(p, f, g)
    oversampled_fft!(p, g)
    resampling!(p, g, out)
    @test out ≈ p * f

    g = zeros(ComplexF64, Ñ); back = zeros(ComplexF64, N)
    resampling_adjoint!(p, fHat, g)
    oversampled_bfft!(p, g)
    resampling_correction_adjoint!(p, g, back)
    @test back ≈ adjoint(p) * fHat

    # the resampling stage as a sparse matrix
    B = sparse(p)
    @test size(B) == (J, prod(Ñ)) && nnz(B) <= J * p.kernel.W^D
    g = randn(rng, ComplexF64, Ñ)
    @test B * vec(g) ≈ resampling!(p, g, zeros(ComplexF64, J))
    @test B' * fHat ≈ vec(resampling_adjoint!(p, fHat, zeros(ComplexF64, Ñ)))

    @test_throws DimensionMismatch resampling!(p, zeros(ComplexF64, N), out)
    @test_throws DimensionMismatch resampling!(p, g, zeros(ComplexF64, J + 1))
    @test_throws DimensionMismatch resampling_correction!(p, zeros(ComplexF64, Ñ), g)
    @test_throws DimensionMismatch resampling_adjoint!(p, fHat, zeros(ComplexF64, N))
  end

  # NFCT and NFST: the oversampled array holds the stored points, the FFT stages are the DCT-I/DST-I
  for P in (NFCTPlan, NFSTPlan), N in ((40,), (17, 22), (9, 10, 8))
    D = length(N); J = 120
    k = rand(rng, D, J) ./ 2
    p = P(k, N; reltol=1e-9)
    Ñ = oversampled_size(p)
    f = randn(rng, N); fHat = randn(rng, J)
    g = zeros(Ñ); out = zeros(J)
    resampling_correction!(p, f, g)
    oversampled_fft!(p, g)
    resampling!(p, g, out)
    @test out ≈ p * f
    g = zeros(Ñ); back = zeros(N)
    resampling_adjoint!(p, fHat, g)
    oversampled_bfft!(p, g)
    resampling_correction_adjoint!(p, g, back)
    @test back ≈ transpose(p) * fHat
    # the FFT stages are full transforms: with arbitrary data they are each other's transpose
    x = randn(rng, Ñ); y = randn(rng, Ñ)
    @test dot(y, oversampled_fft!(p, copy(x))) ≈ dot(oversampled_bfft!(p, copy(y)), x)
    @test_throws DimensionMismatch resampling!(p, zeros(N), out)
  end

  # the stage interface needs a plan without batch dimensions
  p = plan_nfft(rand(rng, 1, 20) .- 0.5, (3, 16); dims=2)
  @test_throws ArgumentError oversampled_size(p)
end
