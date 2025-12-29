#=
Test sampling density compensation
=#
using NFFT: plan_nfft
using NFFTTools: sdc
import NFFT # LINEAR, FULL, TENSOR, POLYNOMIAL
using Test: @test, @testset

@testset "Sampling Density" begin

# create a 10×8 grid of unit spaced sampling points
N = (10, 8)
T = Float32
g = N -> @. T((0:(N-1)) / N - 0.5)
nodes = hcat([[x; y] for x in g(N[1]), y in g(N[2])]...)

for pre in [NFFT.LINEAR, NFFT.FULL, NFFT.TENSOR, NFFT.POLYNOMIAL]
  # approximate the density weights
  p = plan_nfft(nodes, N, m = 5, σ = 2.0, precompute=pre)
  weights = sdc(p, iters = 10)

  @test T == eltype(weights)
  @test all(isreal, weights)
  @test all(>(0), weights)
  @test all(≈(1/prod(N)), weights)
end

end # @testset
