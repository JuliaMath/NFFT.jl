using NFFTTools: calculateToeplitzKernel, calculateToeplitzKernel!, convolveToeplitzKernel!
using FFTW

@testset "Toeplitz" begin
  rng = Xoshiro(1)

  # kernel against the explicit evaluation (2D, Float64)
  for Nx in (32, 33, 64)
    trj = rand(rng, 2, 1000) .- 0.5
    Kx = NFFTTools.calculateToeplitzKernel_explicit((Nx, Nx), trj)
    Ka = calculateToeplitzKernel((Nx, Nx), trj)
    @test Ka isa Matrix{ComplexF64}
    @test Ka ≈ Kx rtol = 1e-7

    p = plan_nfft(rand(rng, 2, 10) .- 0.5, (2Nx, 2Nx))
    Kb = similar(Kx)
    calculateToeplitzKernel!(Kb, p, trj, plan_fft(Kb; flags = FFTW.ESTIMATE))
    @test Kb ≈ Kx rtol = 1e-7
  end

  # Float32, square, rectangular and 3D
  for shape in ((32, 32), (32, 33), (16, 16, 16))
    trj = Float32.(rand(rng, length(shape), 2000) .- 0.5)
    Kx = NFFTTools.calculateToeplitzKernel_explicit(shape, trj)
    Ka = calculateToeplitzKernel(shape, trj; reltol = 1e-5)
    @test Ka isa Array{ComplexF32,length(shape)}
    @test size(Ka) == 2 .* shape
    @test Ka ≈ Kx rtol = 1e-4
  end

  # the Toeplitz product equals Aᴴ A
  Nx = 32
  trj = Float32.(rand(rng, 2, 5000) .- 0.5)
  Ka = calculateToeplitzKernel((Nx, Nx), trj; reltol = 1e-5)
  x = randn(rng, ComplexF32, Nx, Nx)
  p = plan_nfft(trj, (Nx, Nx); reltol = 1e-5)
  xN = adjoint(p) * (p * x)
  x0 = copy(x)
  @test convolveToeplitzKernel!(copy(x), Ka) ≈ xN rtol = 1e-4   # default plans (FFTL)
  xOS1, xOS2 = similar(Ka), similar(Ka)
  fftplan = NFFTTools.plan_fft(xOS1)
  ifftplan = NFFTTools.plan_ifft(xOS1)
  convolveToeplitzKernel!(x, Ka, fftplan, ifftplan, xOS1, xOS2)
  @test x ≈ xN rtol = 1e-4
  x .= x0
  convolveToeplitzKernel!(x, Ka, fftplan, ifftplan, xOS1, xOS2)
  @test (@allocated convolveToeplitzKernel!(x, Ka, fftplan, ifftplan, xOS1, xOS2)) == 0
  # plans of another FFT package (FFTW)
  x .= x0
  convolveToeplitzKernel!(x, Ka, plan_fft(xOS1; flags = FFTW.MEASURE), plan_ifft(xOS1; flags = FFTW.MEASURE), xOS1, xOS2)
  @test x ≈ xN rtol = 1e-4

  # with weights the Toeplitz product equals Aᴴ diag(w) A
  w = rand(rng, Float32, size(trj, 2))
  Kw = calculateToeplitzKernel((Nx, Nx), trj; weights = w, reltol = 1e-5)
  @test convolveToeplitzKernel!(copy(x0), Kw) ≈ adjoint(p) * (w .* (p * x0)) rtol = 1e-4
  pw = plan_nfft(trj, (2Nx, 2Nx); reltol = 1e-5)
  Kb = similar(Kw)
  calculateToeplitzKernel!(Kb, pw, trj, NFFTTools.plan_fft(Kb); weights = w)
  @test Kb ≈ Kw rtol = 1e-4
  @test_throws DimensionMismatch calculateToeplitzKernel((Nx, Nx), trj; weights = w[1:10])
end
