# NFFT.jl-specific behaviour of plans: backends, options, errors, copies.

@testset "Plan" begin

  @testset "backend" begin
    @test NFFT.backend() isa NFFTBackend
    AbstractNFFTs.set_active_backend!(missing)
    @test ismissing(AbstractNFFTs.active_backend())
    with(nfft_backend => NFFT.backend()) do
      @test AbstractNFFTs.active_backend() isa NFFTBackend
    end
    @test ismissing(AbstractNFFTs.active_backend())
    NFFT.activate!()
    @test AbstractNFFTs.active_backend() isa NFFTBackend
  end

  rng = Xoshiro(7)
  N = (24, 20); J = 300
  k = rand(rng, 2, J) .- 0.5
  f = randn(rng, ComplexF64, N); fHat = randn(rng, ComplexF64, J)

  @testset "argument errors" begin
    @test_throws ArgumentError plan_nfft(zeros(1, 4), (2, 2))               # node dimension
    @test_throws ArgumentError plan_nfft([-0.6 0.3; 0.2 -0.1], (4, 4))      # node out of range
    @test_throws ArgumentError plan_nfft([-0.3 0.3; 0.3 NaN], (4, 4))       # NaN node
    @test_throws ArgumentError plan_nfft(k, N; m=4)                          # unknown option
    @test_throws ArgumentError plan_nfft(k, N; reltol=-1)
    @test_throws ArgumentError plan_nfft(k, N; dims=2:3)
    @test_throws ArgumentError plan_nfft(k, N; window=CustomWindow((z, W) -> exp(-z^2)))  # needs W, σ
    @test_throws ArgumentError plan_nfft(k, N; σ=0.8)                        # grid smaller than N
    @test_throws ArgumentError plan_nfft(k, N; W=1)
    p = plan_nfft(k, N)
    @test_throws ArgumentError nodes!(p, [0.7 0.1; 0.1 0.1])
    @test_throws DimensionMismatch nodes!(p, rand(3, 10) .- 0.5)
    @test_throws DimensionMismatch mul!(zeros(ComplexF64, J + 1), p, f)
    @test_throws DimensionMismatch mul!(zeros(ComplexF64, N .+ 1), adjoint(p), fHat)
  end

  @testset "nodes at ±1/2 and given as ranges" begin
    p = plan_nfft(range(-0.5, 0.5; length=50), 30)
    q = plan_nfft(collect(range(-0.5, 0.5; length=50)), 30)
    g = randn(rng, ComplexF64, 30)
    @test p * g ≈ q * g
    @test nodes(p) == range(-0.5, 0.5; length=50)'
  end

  @testset "options are respected" begin
    p = plan_nfft(k, N; window=KaiserBessel(), W=9, σ=1.75)
    @test p.kernel.window == KaiserBessel() && p.kernel.W == 9 && p.kernel.σ == 1.75
    p = plan_nfft(k, N; window=ExpSemicircle())
    @test p.kernel.window == ExpSemicircle()
    for tol in (1e-4, 1e-8, 1e-12)
      @test accuracy(plan_nfft(k, N; reltol=tol)) <= tol
    end
    # the most accurate kernel is reachable, anything below it an error
    best = minimum(NFFT.predicted_error(e, N, Float64) for e in NFFT.kernel_candidates(:auto, nothing, nothing, Float64))
    @test best < 1e-13
    @test accuracy(plan_nfft(k, N; reltol=best)) == best
    @test_throws ArgumentError plan_nfft(k, N; reltol=best / 2)
    @test_throws ArgumentError plan_nfft(Float32.(k), N; reltol=1e-9)
    @test_throws ArgumentError plan_nfft(k, N; reltol=1e-14, W=6)
    @test plan_nfft(k, N; reltol=1e-14, W=6, σ=2.0).kernel.W == 6      # W and σ fixed: as given
    @test_throws ArgumentError plan_nnfft(k, 4 .* k; reltol=1e-17)
    @test plan_nfft(Float32.(k), N).options.reltol == default_reltol(Float32)
    # beyond the table: Kaiser–Bessel with an analytic β (explicit W, or automatic for wider types)
    p = plan_nfft(k, N; W=20, σ=2.0)
    @test p.kernel.window == KaiserBessel() && p.kernel.β ≈ π * 20 * 0.75
    @test norm(p * f - NDFTPlan(k, N) * f) / norm(NDFTPlan(k, N) * f) < 1e-13
    @test eltype(plan_nfft(k, N; eltype=Float32)) == ComplexF32
    # measure: one of the cheapest candidates of the cost model, same transform
    p = plan_nfft(k, N; reltol=1e-8, measure=true)
    cands = first.(NFFT.choose_kernels(1e-8, N, J, Float64; n=NFFT.MEASURE_CANDIDATES))
    @test p.kernel in cands
    @test accuracy(p) <= 1e-8
    @test norm(p * f - plan_nfft(k, N; reltol=1e-12) * f) / norm(p * f) < 1e-7
    q = plan_nfst(rand(rng, 2, 50) ./ 2, (12, 10); reltol=1e-6, measure=true, W=8)
    @test q.kernel.W == 8
    # cost model: fitted on this machine, or set per storage and element type
    c = NFFT.calibrate_cost(; densities=(0.1, 1), W=4:5, σ=(1.5, 2.0), dims=1:1, sizes=(((2^10,), (2^12,)),), nrep=1, set=false)
    @test all(>=(0), c[1])
    @test length(c[1]) == 4 && all(isfinite, c[1])
    fftonly = Dict(D => (0.0, 0.0, 1.0, 0.0) for D in 1:3)
    NFFT.set_cost_model!(Array, Float64, fftonly)
    e = first(NFFT.choose_kernels(1e-6, N, J, Float64))[1]
    NFFT.set_cost_model!(Array, Float64, nothing)
    @test e.σ <= 1.25   # the smallest grid (σ = 1.2 and 1.25 round to the same size)
    # Wmax (the C library's compiled widths): a narrower window on a larger grid that still reaches
    # reltol (3D Float32 at 1e-4: W = 9 without the bound), or an error
    k3 = rand(rng, 3, 60) .- 0.5; f3 = randn(rng, ComplexF32, 12, 9, 6); v3 = randn(rng, ComplexF32, 60)
    @test first(NFFT.choose_kernels(1e-4, (12, 9, 6), 60, Float32))[1].W > 8
    for tol in (1e-4, 1e-5)
      e, err = first(NFFT.choose_kernels(tol, (12, 9, 6), 60, Float32; Wmax=8))
      @test e.W <= 8 && err <= tol
      p = plan_nfft(Float32.(k3), (12, 9, 6); window=e.window, W=e.W, σ=e.σ)
      ref = NDFTPlan(Float64.(Float32.(k3)), (12, 9, 6))
      @test relerr(p * f3, ref * ComplexF64.(f3)) <= tol
      @test relerr(adjoint(p) * v3, adjoint(ref) * ComplexF64.(v3)) <= tol
    end
    @test_throws ArgumentError NFFT.choose_kernels(1e-10, (12, 9, 6), 60, Float64; Wmax=8)
    @test NFFT.cost_model(Array, Float64) === NFFT.COST_COEF
    # a custom window with explicit width and oversampling
    gauss = CustomWindow((z, W) -> abs(z) < W / 2 ? exp(-(2z / W)^2 * 9) : 0.0)
    p = plan_nfft(k, N; window=gauss, W=8, σ=2.0)
    @test isnan(accuracy(p))
    @test norm(p * f - plan_nfft(k, N; reltol=1e-10) * f) / norm(p * f) < 1e-2
    # gridsize: any grid of at least σN points, also with factors 7, 11 and odd
    ref = NDFTPlan(k, N)
    for Ñ in ((56, 44), (49, 33))
      p = plan_nfft(k, N; window=KaiserBessel(), W=10, σ=1.6, gridsize=Ñ)
      @test p.core.Ñ == Ñ
      @test norm(p * f - ref * f) / norm(ref * f) < 1e-7
      @test norm(adjoint(p) * fHat - adjoint(ref) * fHat) / norm(adjoint(ref) * fHat) < 1e-7
    end
    # Ñ + W a multiple of 8: the padded grid gets 4 extra points (padded_size)
    p = plan_nfft(k, N; window=KaiserBessel(), W=10, σ=1.6, gridsize=(54, 46), nthreads=3)
    @test size(p.core.gp) == (68, 56)
    @test norm(p * f - ref * f) / norm(ref * f) < 1e-7
    @test norm(adjoint(p) * fHat - adjoint(ref) * fHat) / norm(adjoint(ref) * fHat) < 1e-7
    k3 = rand(rng, 3, 400) .- 0.5; f3 = randn(rng, ComplexF64, 12, 10, 14); v3 = randn(rng, ComplexF64, 400)
    ref3 = NDFTPlan(k3, (12, 10, 14))
    p = plan_nfft(k3, (12, 10, 14); window=KaiserBessel(), W=10, σ=2.0, gridsize=(30, 22, 30), nthreads=3)
    @test size(p.core.gp) == (44, 36, 40)
    @test norm(p * f3 - ref3 * f3) / norm(ref3 * f3) < 1e-7
    @test norm(adjoint(p) * v3 - adjoint(ref3) * v3) / norm(adjoint(ref3) * v3) < 1e-7
    p = plan_nfft(k, N; reltol=1e-6, gridsize=77)
    @test p.core.Ñ == (77, 77)
    @test norm(p * f - ref * f) / norm(ref * f) < 1e-6
    nodes!(p, k[:, 1:100])
    @test p.core.Ñ == (77, 77)
    fb = randn(rng, ComplexF64, N..., 3)
    pb = plan_nfft(k, (N..., 3); dims=1:2, W=10, σ=2.0, gridsize=(56, 44))
    @test norm(pb * fb - plan_nfft(k, (N..., 3); dims=1:2, W=10, σ=2.0) * fb) / norm(pb * fb) < 1e-8
    @test_throws ArgumentError plan_nfft(k, N; W=10, σ=2.0, gridsize=(47, 44))   # below σN
    @test_throws ArgumentError plan_nfft(k, N; W=10, σ=2.0, gridsize=(56, 44, 3))
    @test_throws ArgumentError plan_nfct(rand(rng, 2, 50) ./ 2, N; gridsize=(56, 44))
  end

  @testset "threads and copies give the same result" begin
    p = plan_nfft(k, N; reltol=1e-9)
    q = plan_nfft(k, N; reltol=1e-9, nthreads=1)
    @test p * f ≈ q * f
    @test adjoint(p) * fHat ≈ adjoint(q) * fHat
    c = copy(p)
    @test c.core.gp !== p.core.gp
    @test c * f == p * f
    @test adjoint(c) * fHat ≈ adjoint(p) * fHat
    # copies used concurrently
    plans = [copy(p) for _ in 1:4]
    out = Vector{Any}(undef, 4)
    Threads.@threads for i in 1:4
      out[i] = plans[i] * f
    end
    @test all(o -> o == p * f, out)
  end

  @testset "FFTW: flags are kept, the global thread count is not changed" begin
    n = NFFT.FFTW.get_num_threads()
    fft = NFFT.FFTWBackend(flags=NFFT.FFTW.MEASURE)
    p = plan_nfft(k, N; fft)
    @test NFFT.FFTW.get_num_threads() == n
    @test copy(p).core.fft.flags == NFFT.FFTW.MEASURE
    for nt in (1, 4)
      q = plan_nfft(k, (N..., 16); dims=1:2, fft, nthreads=nt)
      cores = q.core isa Vector ? q.core : [q.core]
      cores[1] isa NFFT.NFFTCore && @test all(c -> c.fft.flags == NFFT.FFTW.MEASURE, cores)
    end
  end

  @testset "show" begin
    p = plan_nfft(k, N)
    @test occursin("NFFTPlan", sprint(show, p))
    @test occursin("Adjoint", sprint(show, adjoint(p)))
  end
end
