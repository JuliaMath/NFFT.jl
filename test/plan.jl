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
    @test accuracy(plan_nfft(k, N; reltol=1e-16)) < 1e-13         # below the floor: the most accurate kernel
    @test eltype(plan_nfft(k, N; eltype=Float32)) == ComplexF32
    # measure: one of the cheapest candidates of the cost model, same transform
    p = plan_nfft(k, N; reltol=1e-8, measure=true)
    cands = first.(NFFT.choose_kernels(1e-8, N, J, Float64; n=NFFT.MEASURE_CANDIDATES))
    @test p.kernel in cands
    @test accuracy(p) <= 1e-8
    @test norm(p * f - plan_nfft(k, N; reltol=1e-12) * f) / norm(p * f) < 1e-7
    q = plan_nfst(rand(rng, 2, 50) ./ 2, (12, 10); measure=true, W=8)
    @test q.kernel.W == 8
    # cost model: fitted on this machine, or set per storage and element type
    c = NFFT.calibrate_cost(; J=(2^10, 2^12), W=4:5, σ=(1.5, 2.0), dims=1:1, sizes=(((2^12,), (2^14,)),), nrep=1, set=false)
    @test length(c[1]) == 4 && all(isfinite, c[1])
    fftonly = Dict(D => (0.0, 0.0, 1.0, 0.0) for D in 1:3)
    NFFT.set_cost_model!(Array, Float64, fftonly)
    e = first(NFFT.choose_kernels(1e-6, N, J, Float64))[1]
    NFFT.set_cost_model!(Array, Float64, nothing)
    @test e.σ <= 1.25   # the smallest grid (σ = 1.2 and 1.25 round to the same size)
    @test NFFT.cost_model(Array, Float64) === NFFT.COST_COEF
    # a custom window with explicit width and oversampling
    gauss = CustomWindow((z, W) -> abs(z) < W / 2 ? exp(-(2z / W)^2 * 9) : 0.0)
    p = plan_nfft(k, N; window=gauss, W=8, σ=2.0)
    @test isnan(accuracy(p))
    @test norm(p * f - plan_nfft(k, N; reltol=1e-10) * f) / norm(p * f) < 1e-2
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
