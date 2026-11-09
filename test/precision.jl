# Double64 and Float128 with the FFT of FFTA.jl: every transform against the direct sum in
# the same type at a tolerance far below Float64, and an unreachable tolerance throws. Kernels of
# width W ≈ 20 compile slowly for Double64, so the sizes stay small and mostly 1D.
using DoubleFloats, Quadmath

@testset "Precision beyond Float64" begin
  modes(I, N) = ntuple(d -> d == 1 ? I[1] - 1 : I[d] - 1 - N[d] ÷ 2, length(N))
  rdirect(k, N, h) = [sum(real(h[I] * cispi(-2 * sum(modes(I, N) .* Tuple(k[:, j])))) for I in CartesianIndices(h))
                      for j in 1:size(k, 2)]
  radjoint(k, N, y) = [sum(y[j] * cispi(2 * sum(modes(I, N) .* Tuple(k[:, j]))) for j in eachindex(y))
                       for I in CartesianIndices((N[1] ÷ 2 + 1, Base.tail(N)...))]
  fft = FFTABackend()

  @testset "$T" for (T, tol, unreachable) in ((Double64, 1e-20, 1e-35), (Float128, 1e-20, 1e-38))
    rng = Xoshiro(5)
    N = (16,); J = 20
    k = T.(rand(rng, 1, J) .- 0.5); kr = abs.(k)
    f = randn(rng, Complex{T}, N); y = randn(rng, Complex{T}, J); yr = real.(y)

    p = plan_nfft(k, N; reltol=tol, fft); q = NDFTPlan(k, N)
    @test p * f isa Vector{Complex{T}}
    @test relerr(p * f, q * f) <= tol
    @test relerr(adjoint(p) * y, adjoint(q) * y) <= tol

    p = plan_rnfft(k, N; reltol=tol, fft); h = randn(rng, Complex{T}, size_in(p))
    @test relerr(p * h, rdirect(k, N, h)) <= tol
    @test relerr(adjoint(p) * yr, radjoint(k, N, yr)) <= tol

    for (P, Q) in ((plan_nfct, NDCTPlan), (plan_nfst, NDSTPlan))
      p = P(kr, N; reltol=tol, fft); q = Q(kr, N); g = randn(rng, T, N)
      @test relerr(p * g, q * g) <= tol
      @test relerr(transpose(p) * yr, transpose(q) * yr) <= tol
    end

    src = T.((rand(rng, 1, J) .- 0.5) .* 8)
    p = plan_nnfft(k, src; reltol=tol, fft); q = NNDFTPlan(k, src)
    @test relerr(p * y, q * y) <= tol
    @test relerr(adjoint(p) * y, adjoint(q) * y) <= tol

    @test_throws ArgumentError plan_nfft(k, N; reltol=unreachable, fft)
    @test_throws ArgumentError plan_rnfft(k, N; reltol=unreachable, fft)
    @test_throws ArgumentError plan_nfct(kr, N; reltol=unreachable, fft)
    @test_throws ArgumentError plan_nfst(kr, N; reltol=unreachable, fft)
    @test_throws ArgumentError plan_nnfft(k, src; reltol=unreachable, fft)
  end

  @testset "Float128 in 2D" begin
    rng = Xoshiro(6)
    N = (10, 8); J = 30
    k = Float128.(rand(rng, 2, J) .- 0.5); f = randn(rng, Complex{Float128}, N)
    p = plan_nfft(k, N; reltol=1e-20, fft)
    @test relerr(p * f, NDFTPlan(k, N) * f) <= 1e-20
  end
end
