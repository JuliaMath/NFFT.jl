#=
Test sampling density compensation
=#
using NFFT: plan_nfft
using NFFTTools: sdc
import NFFT # LINEAR, FULL, TENSOR, POLYNOMIAL
import NFFTTools # sdc!
using Test: @test, @testset

@testset "Sampling Density" begin

# create a 9×8 grid of unit spaced sampling points
N = (9, 8)
T = Float32
fun = N -> @. T((0:(N-1)) / N - 0.5) # 1D grid
nodes = hcat([[x; y] for x in fun(N[1]), y in fun(N[2])]...)

for pre in [NFFT.LINEAR, NFFT.FULL, NFFT.TENSOR, NFFT.POLYNOMIAL]
  # approximate the density weights
  p = plan_nfft(nodes, N, m = 5, σ = 2.0, precompute=pre)
  @show @allocated weights = sdc(p; iters = 10)

  @test T == eltype(weights)
  @test isreal(weights)
  @test all(>(0), weights)
  @test all(≈(1/prod(N)), weights)

  # Test the version with provided work buffers
  w2 = deepcopy(weights)
  fill!(w2, one(T))
  sdc(p; iters = 10,
      weights = w2,
      weights_tmp = similar(w2),
      workg = Array{real(eltype(p.tmpVec))}(undef, p.Ñ),
      workf = similar(p.tmpVec, Complex{T}, p.J),
      workv = similar(p.tmpVec, Complex{T}, p.N),
  )
  @test w2 == weights

  # pre-allocate
  weights_tmp = similar(w2)
  workg = Array{real(eltype(p.tmpVec))}(undef, p.Ñ)
  workf = similar(p.tmpVec, Complex{T}, p.J)
  workv = similar(p.tmpVec, Complex{T}, p.N)
  fill!(w2, one(T))

# ideally the following test should be non-allocating!?
# @test 0 == @allocated
  @show @allocated NFFTTools.sdc!(p, 10,
      w2,
      weights_tmp,
      workg,
      workf,
      workv,
  )
  @test w2 == weights

end # pre

end # @testset
