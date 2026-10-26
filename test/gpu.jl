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

    # the adjoint resampling by tiles (2D, 3D, if its buffers fit into local memory) has no
    # atomics: repeated calls agree exactly
    if p.core.nodes.tiles !== nothing
      @test Array(adjoint(p) * arrayType(fHat)) == x
    end

    # storage defaults to the array type of the nodes
    q = plan_nfft(arrayType(k), N; reltol)
    @test relerr(Array(q * arrayType(f)), y) < 100 * eps(T)
  end

  @testset "batch $T, N = $N, dims = $dims" for T in floatTypes,
      (N, dims) in [((4, 256), 2:2), ((256, 4), 1:1), ((3, 32, 32), 2:3), ((24, 24, 2), 1:2), ((2, 12, 12, 12), 2:4)]
    Random.seed!(2)
    Dd = length(dims)
    J = 500
    k = rand(T, Dd, J) .- T(1 // 2)
    reltol = T == Float32 ? 1e-5 : 1e-9
    p = plan_nfft(k, N; dims, reltol, storage=arrayType)
    pcpu = plan_nfft(k, N; dims, reltol)
    f = randn(Complex{T}, N)
    fHat = randn(Complex{T}, size_out(pcpu))
    @test relerr(Array(p * arrayType(f)), pcpu * f) < 100 * eps(T)
    @test relerr(Array(adjoint(p) * arrayType(fHat)), adjoint(pcpu) * fHat) < 100 * eps(T)
  end

  # real transforms against the CPU plans: RNFFT (real values at the nodes), NFCT, NFST (real data,
  # nodes in [0, 1/2]); with batch dimensions
  @testset "$name $T, N = $N, dims = $dims" for name in ("rnfft", "nfct", "nfst"), T in floatTypes,
      (N, dims) in [((256,), 1:1), ((32, 32), 1:2), ((12, 12, 12), 1:3), ((3, 32, 32), 2:3), ((24, 20, 2), 1:2)]
    Random.seed!(3)
    Dd = length(dims)
    J = 600
    reltol = T == Float32 ? 1e-5 : 1e-9
    if name == "rnfft"
      k = rand(T, Dd, J) .- T(1 // 2)
      p = plan_rnfft(k, N; dims, reltol, storage=arrayType)
      pcpu = plan_rnfft(k, N; dims, reltol)
      x = randn(Complex{T}, size_in(pcpu))
      y = randn(T, size_out(pcpu))
      back = adjoint
    else
      k = rand(T, Dd, J) ./ 2
      plan = name == "nfct" ? plan_nfct : plan_nfst
      p = plan(k, N; dims, reltol, storage=arrayType)
      pcpu = plan(k, N; dims, reltol)
      x = randn(T, N)
      y = randn(T, size_out(pcpu))
      back = transpose
    end
    @test relerr(Array(p * arrayType(x)), pcpu * x) < 100 * eps(T)
    @test relerr(Array(back(p) * arrayType(y)), back(pcpu) * y) < 100 * eps(T)
  end

  @testset "nnfft $T, D = $D" for T in floatTypes, D in 1:3
    Random.seed!(4)
    y = (rand(T, D, 200) .- T(0.3)) .* 6 .+ 1
    k = (rand(T, D, 300) .- T(0.6)) .* 12
    reltol = T == Float32 ? 1e-4 : 1e-9
    p = plan_nnfft(k, y; reltol, storage=arrayType)
    pcpu = plan_nnfft(k, y; reltol)
    f = randn(Complex{T}, 200); v = randn(Complex{T}, 300)
    @test relerr(Array(p * arrayType(f)), pcpu * f) < 100 * eps(T) * pcpu.outer.amp
    @test relerr(Array(adjoint(p) * arrayType(v)), adjoint(pcpu) * v) < 100 * eps(T) * pcpu.outer.amp
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
