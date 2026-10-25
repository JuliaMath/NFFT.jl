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

    @test_throws DimensionMismatch resampling!(p, zeros(ComplexF64, N), out)
    @test_throws DimensionMismatch resampling!(p, g, zeros(ComplexF64, J + 1))
    @test_throws DimensionMismatch resampling_correction!(p, zeros(ComplexF64, Ñ), g)
    @test_throws DimensionMismatch resampling_adjoint!(p, fHat, zeros(ComplexF64, N))
  end

  # the stage interface needs a plan without batch dimensions
  p = plan_nfft(rand(rng, 1, 20) .- 0.5, (3, 16); dims=2)
  @test_throws ArgumentError oversampled_size(p)
end
