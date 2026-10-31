# The FFT backend is exchangeable: NFFT and NNFFT plans with FFTA.jl (pure Julia) give the results
# of the FFTW plans, and the cores hold whatever passes the backend makes.

@testset "FFT backend FFTA" begin
  rng = Xoshiro(21)
  ffta_plan(k, N; kwargs...) = plan_nfft(k, N; fft=FFTABackend(), kwargs...)
  @testset "contract" begin
    contract_tests(ffta_plan; T=Float64, tols=(1e-4, 1e-9), sizes=((64,), (31, 32), (12, 13, 10)), rng)
  end

  @testset "against FFTW, $N" for N in ((40,), (17, 22), (9, 10, 8))
    D = length(N); J = 200
    k = rand(rng, D, J) .- 0.5
    p = plan_nfft(k, N; reltol=1e-9, fft=FFTABackend())
    q = plan_nfft(k, N; reltol=1e-9)
    @test !(p.core.fwd isa typeof(q.core.fwd))
    f = randn(rng, ComplexF64, N); fHat = randn(rng, ComplexF64, J)
    @test p * f ≈ q * f
    @test adjoint(p) * fHat ≈ adjoint(q) * fHat
    @test copy(p) * f ≈ q * f
    # stage interface: the FFT stages of the backend
    g = randn(rng, ComplexF64, oversampled_size(p))
    @test oversampled_fft!(p, copy(g)) ≈ oversampled_fft!(q, copy(g))
    @test oversampled_bfft!(p, copy(g)) ≈ oversampled_bfft!(q, copy(g))
  end

  @testset "batch, dims = $dims" for (N, dims) in (((3, 20, 22), 2:3), ((24, 26, 4), 1:2), ((50, 6), 1))
    k = rand(rng, length(dims), 100) .- 0.5
    p = plan_nfft(k, N; dims, reltol=1e-9, fft=FFTABackend())
    q = plan_nfft(k, N; dims, reltol=1e-9)
    f = randn(rng, ComplexF64, N)
    @test p * f ≈ q * f
    @test adjoint(p) * (q * f) ≈ adjoint(q) * (q * f)
  end

  @testset "NNFFT" begin
    k = rand(rng, 2, 150) .- 0.5; y = rand(rng, 2, 120) .- 0.5
    p = plan_nnfft(k, y; reltol=1e-9, fft=FFTABackend())
    q = plan_nnfft(k, y; reltol=1e-9)
    f = randn(rng, ComplexF64, 120)
    @test p * f ≈ q * f
  end

  # the real transforms through complex FFTs of the same length
  @testset "RNFFT, NFCT, NFST" begin
    k = rand(rng, 2, 50) ./ 2; y = randn(rng, 50)
    p = plan_rnfft(k .- 0.25, (16, 12); reltol=1e-9, fft=FFTABackend()); q = plan_rnfft(k .- 0.25, (16, 12); reltol=1e-9)
    h = randn(rng, ComplexF64, size_in(p))
    @test p * h ≈ q * h
    @test adjoint(p) * y ≈ adjoint(q) * y
    g = randn(rng, oversampled_size(p)); ĝ = randn(rng, ComplexF64, spectrum_size(p))
    @test oversampled_bfft!(p, g, similar(ĝ)) ≈ oversampled_bfft!(q, g, similar(ĝ))
    @test oversampled_fft!(p, ĝ, similar(g)) ≈ oversampled_fft!(q, ĝ, similar(g))
    for P in (plan_nfct, plan_nfst)
      p = P(k, (16, 12); reltol=1e-9, fft=FFTABackend()); q = P(k, (16, 12); reltol=1e-9)
      f = randn(rng, 16, 12)
      @test p * f ≈ q * f
      @test transpose(p) * y ≈ transpose(q) * y
    end
  end
end
