# Dimensions above 3 (D = 4 and 5) against the direct sums: NFFT, NNFFT, RNFFT, NFCT and NFST,
# transform and adjoint (transpose), Float64 and Float32, batches over dims, several threads.
@testset "D > 3" begin
  rng = Xoshiro(41)
  J = 50
  # RNFFT: f̂_j = Re Σ h_n exp(−2πi n·k_j), n₁ = 0…N₁÷2, the other dimensions centred
  rmodes(I, N) = ntuple(d -> d == 1 ? I[1] - 1 : I[d] - 1 - N[d] ÷ 2, length(N))
  rdirect(k, N, h) = [sum(real(h[I] * cispi(-2 * sum(rmodes(I, N) .* Tuple(k[:, j])))) for I in CartesianIndices(h))
                      for j in 1:size(k, 2)]
  radjoint(k, N, y) = [sum(y[j] * cispi(2 * sum(rmodes(I, N) .* Tuple(k[:, j]))) for j in eachindex(y))
                       for I in CartesianIndices((N[1] ÷ 2 + 1, Base.tail(N)...))]

  @testset "$(length(N))D $T nthreads=$nt" for (N, T, nt) in (((6, 6, 6, 6), Float64, 1), ((6, 6, 6, 6), Float32, 1),
                                                             ((6, 5, 6, 4), Float64, 3), ((4, 4, 4, 4, 4), Float64, 1))
    D = length(N); reltol = T == Float64 ? 1e-8 : 1e-4
    k = rand(rng, D, J) .- 0.5; kT = T.(k); k64 = Float64.(kT)
    tol(p) = allowed(accuracy(p), T, N)

    p = plan_nfft(kT, N; reltol, nthreads=nt)
    f = randn(rng, Complex{T}, N); v = randn(rng, Complex{T}, J)
    @test relerr(p * f, NDFTPlan(k64, N) * f) < tol(p)
    @test relerr(adjoint(p) * v, adjoint(NDFTPlan(k64, N)) * v) < tol(p)

    y = 4 .* (rand(rng, D, J) .- 0.5); yT = T.(y)
    p = plan_nnfft(kT, yT; reltol, nthreads=nt)
    @test relerr(p * v, NNDFTPlan(k64, Float64.(yT)) * v) < allowed(accuracy(p), T, (J,))
    @test relerr(adjoint(p) * v, adjoint(NNDFTPlan(k64, Float64.(yT))) * v) < allowed(accuracy(p), T, (J,))

    p = plan_rnfft(kT, N; reltol, nthreads=nt)
    h = randn(rng, Complex{T}, size_in(p)); w = randn(rng, T, J)
    @test relerr(p * h, rdirect(k64, N, h)) < tol(p)
    @test relerr(adjoint(p) * w, radjoint(k64, N, w)) < tol(p)

    kr = T.(rand(rng, D, J) ./ 2); kr64 = Float64.(kr); fr = randn(rng, T, N)
    for (plan, Pd) in ((plan_nfct, NDCTPlan), (plan_nfst, NDSTPlan))
      p = plan(kr, N; reltol, nthreads=nt); pd = Pd(kr64, N)
      @test relerr(p * fr, pd * fr) < tol(p)
      @test relerr(transpose(p) * w, transpose(pd) * w) < tol(p)
    end
  end

  @testset "batch over dims, 4D" begin
    N = (5, 6, 4, 6); L = 3
    k = rand(rng, 4, J) .- 0.5
    p = plan_nfft(k, (N..., L); dims=1:4, reltol=1e-8)
    f = randn(rng, ComplexF64, N..., L); v = randn(rng, ComplexF64, J, L)
    pd = NDFTPlan(k, N)
    for l in 1:L
      @test relerr((p * f)[:, l], pd * f[:, :, :, :, l]) < allowed(accuracy(p), Float64, N)
      @test relerr((adjoint(p) * v)[:, :, :, :, l], adjoint(pd) * v[:, l]) < allowed(accuracy(p), Float64, N)
    end
    # the NFFT dimensions in the middle
    p = plan_nfft(k, (L, N...); dims=2:5, reltol=1e-8)
    g = randn(rng, ComplexF64, L, N...)
    @test relerr((p * g)[1, :], pd * g[1, :, :, :, :]) < allowed(accuracy(p), Float64, N)
  end
end
