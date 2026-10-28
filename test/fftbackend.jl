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

  # the real transforms use FFTW's real and real-to-real transforms
  k = rand(rng, 2, 50) ./ 2
  @test_throws ArgumentError plan_rnfft(k .- 0.25, (16, 16); fft=FFTABackend())
  @test_throws ArgumentError plan_nfct(k, (16, 16); fft=FFTABackend())
  @test_throws ArgumentError plan_nfst(k, (16, 16); fft=FFTABackend())
end
