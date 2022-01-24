using Test
using NFFT
using FFTW
using BenchmarkTools

@testset "Toeplitz" begin

Nx = 32
## calculateToeplitzKernel vs calculateToeplitzKernel_explicit (2D, Float64)
for Nx ∈ [32, 33, 64]
    trj = rand(2, 1000) .- 0.5
    Kx = NFFTTools.calculateToeplitzKernel_explicit((Nx, Nx), trj)
    Ka = calculateToeplitzKernel((Nx, Nx), trj, m=4, σ=2)

    @test typeof(Kx) === Matrix{ComplexF64}
    @test typeof(Ka) === Matrix{ComplexF64}
    @test Ka ≈ Kx rtol = 1e-6
end

## calculateToeplitzKernel with less-allocating constructor vs calculateToeplitzKernel_explicit (2D, Float64)
for Nx ∈ [32, 33, 64]
    trj = rand(2, 1000) .- 0.5
    Kx = NFFTTools.calculateToeplitzKernel_explicit((Nx, Nx), trj)

    p = NFFTPlan(trj, (2Nx, 2Nx))
    Ka = similar(Kx)
    fftplan = plan_fft(Ka; flags = FFTW.ESTIMATE)
    calculateToeplitzKernel!(Ka, p, trj, fftplan)

    @test typeof(Kx) === Matrix{ComplexF64}
    @test typeof(Ka) === Matrix{ComplexF64}
    @test Ka ≈ Kx rtol = 1e-6
end

## calculateToeplitzKernel vs calculateToeplitzKernel_explicit (2D, Float32)
Nx = 32
trj = Float32.(rand(2, 10000) .- 0.5)
Kx = NFFTTools.calculateToeplitzKernel_explicit((Nx, Nx), trj)
Ka = calculateToeplitzKernel((Nx, Nx), trj, m=4, σ=2)

@test typeof(Kx) === Matrix{ComplexF32}
@test typeof(Ka) === Matrix{ComplexF32}
@test Ka ≈ Kx rtol = 1e-5


## compare to the NFFT
x = randn(ComplexF32, Nx, Nx)
xN = nfft_adjoint(trj, (Nx, Nx), nfft(trj, x))

xOS1 = similar(Ka)
xOS2 = similar(Ka)

FFTW.set_num_threads(1)
fftplan = plan_fft(xOS1; flags = FFTW.MEASURE)
ifftplan = plan_ifft(xOS1; flags = FFTW.MEASURE)
convolveToeplitzKernel!(x, Ka, fftplan, ifftplan, xOS1, xOS2)
@test x ≈ xN rtol = 1e-5

## test that convolveToeplitzKernel! with all arguments is non-allocating (only true with FFTW.set_num_threads(1))
bm = @benchmark convolveToeplitzKernel!($x, $Ka, $fftplan, $ifftplan, $xOS1, $xOS2)
@test bm.allocs == 0

## calculateToeplitzKernel vs calculateToeplitzKernel_explicit  (2D-rectangular, Float32)
Nx = 32
Ny = 33
trj = Float32.(rand(2, 1000) .- 0.5)
Kx = NFFTTools.calculateToeplitzKernel_explicit((Nx, Ny), trj)
Ka = calculateToeplitzKernel((Nx, Ny), trj, m=4, σ=2)

@test typeof(Kx) === Matrix{ComplexF32}
@test typeof(Ka) === Matrix{ComplexF32}
@test size(Ka) == (2Nx, 2Ny)
@test Ka ≈ Kx rtol = 1e-5

## calculateToeplitzKernel vs calculateToeplitzKernel_explicit 3D, Float32
Nx = 32
trj = Float32.(rand(3, 1000) .- 0.5)
Kx = NFFTTools.calculateToeplitzKernel_explicit((Nx, Nx, Nx), trj)
Ka = calculateToeplitzKernel((Nx, Nx, Nx), trj, m=4, σ=2)

@test typeof(Kx) === Array{ComplexF32,3}
@test typeof(Ka) === Array{ComplexF32,3}
@test Ka ≈ Kx rtol = 1e-5

end