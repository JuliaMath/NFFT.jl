# GPU tests. Included by test/gpu/<backend>.jl, which loads the GPU package and defines
# `arrayTypes` (GPU array types) and `floatTypes` (precisions the device supports).

using NFFT, NFFTTools, Test, Random, LinearAlgebra
using NFFT: NDFTPlan

relerr(a, b) = norm(a - b) / norm(b)

@testset "GPU $arrayType" for arrayType in arrayTypes
  @testset "$T, N = $N" for T in floatTypes, N in [(256,), (32, 32), (12, 12, 12)]
    Random.seed!(1)
    D = length(N)
    J = prod(N)
    k = rand(T, D, J) .- T(1 // 2)
    reltol = T == Float32 ? 1e-5 : 1e-9

    p = plan_nfft(k, N; reltol, storage=arrayType)
    pcpu = plan_nfft(k, N; reltol)
    pndft = NDFTPlan(k, N)

    f = randn(Complex{T}, N)
    fHat = randn(Complex{T}, J)

    # against the exact transform at the accuracy the plan promises, and against the CPU plan
    # with the same parameters at the rounding level
    y = Array(p * arrayType(f))
    @test y isa Vector{Complex{T}}
    @test relerr(y, pndft * f) < 10 * accuracy(p)
    @test relerr(y, pcpu * f) < 100 * eps(T)

    x = Array(adjoint(p) * arrayType(fHat))
    @test x isa Array{Complex{T},D}
    @test relerr(x, adjoint(pndft) * fHat) < 10 * accuracy(p)
    @test relerr(x, adjoint(pcpu) * fHat) < 100 * eps(T)

    # storage defaults to the array type of the nodes
    q = plan_nfft(arrayType(k), N; reltol)
    @test relerr(Array(q * arrayType(f)), y) < 100 * eps(T)
  end

  @testset "sdc" begin
    # equispaced nodes: all weights equal 1/|N|
    N = (9, 8)
    T = first(floatTypes)
    grid(n) = T.((0:n-1) / n .- 1 // 2)
    nodes = reduce(hcat, [[x; y] for x in grid(N[1]), y in grid(N[2])])
    p = plan_nfft(nodes, N; reltol = T == Float32 ? 1e-5 : 1e-8, storage=arrayType)
    weights = Array(sdc(p; iters=10))
    @test weights ≈ fill(T(1 / prod(N)), length(weights)) rtol = 1e-3
  end
end
