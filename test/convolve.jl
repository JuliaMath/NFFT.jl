#=
Test convolution (interpolation) and its adjoint in isolation

One must be careful with Float16 output arrays here,
because plan.windowLinInterp has values above 1e8
that exceed the 6e4 maximum of Float16.

FFTW does not support Float16 plans as of 2026-01-01.
=#

using NFFT: plan_nfft
using AbstractNFFTs: convolve!, convolve_transpose!
import NFFT # LINEAR, FULL, TENSOR, POLYNOMIAL
using Test: @test, @testset

# determine worst precision of two types
function worst(T1::Type{<:Number}, T2::Type{<:Number})
    return promote_type(T1, T2) == T1 ? T2 : T1
end
worst(a, b, c) = worst(worst(a, b), c)


@testset "convolve!" begin

  N = (9, 8) # 9×8 grid of equally spaced sampling points
  Tb = BigFloat
  fun = N -> @. Tb((0:(N-1)) / N - 0.5) # 1D grid
  nodes = hcat([[x; y] for x in fun(N[1]), y in fun(N[2])]...)

  T16 = Complex{Float16}
  T32 = Complex{Float32}
  T64 = Complex{Float64}
  @test worst(T16, T64) == T16

  J = prod(N) # for f
  Ñ = 2 .* N # for g

  for pre in [NFFT.LINEAR, NFFT.FULL, NFFT.TENSOR, NFFT.POLYNOMIAL]
    p32 = plan_nfft(Float32.(nodes), N, m = 5, σ = 2.0, precompute=pre)
    p64 = plan_nfft(Float64.(nodes), N, m = 5, σ = 2.0, precompute=pre)

    @assert Ñ == p64.Ñ
    f16 = Vector{T16}(undef, J)

    g64 = rand(T64, Ñ)
    g32 = T32.(g64)
    g16 = T16.(g64)
    @test g32 ≈ g64
    @test g16 ≈ g64

    fff = Vector{T64}(undef, J) # ground-truth reference
    @test fff == convolve!(p64, g64, fff)

    fmax = maximum(abs, fff) # need to avoid avoid overflow with T16

    for Tg in (T16, T32, T64), Tf in (T32, T64)
      f = Vector{Tf}(undef, J)
      @test f == convolve!(p64, Tg.(g64), f)
      T = worst(Tg, Tf)
      @test f/fmax ≈ T.(fff/fmax)

      @test f == convolve!(p32, Tg.(g64), f)
      T = worst(Tg, Tf, T32)
      @test f/fmax ≈ T.(fff/fmax)
    end

    # Float16 test with careful scaling:
    @test f16 == convolve!(p64, g64/fmax, f16)
    @test f16 ≈ fff/fmax
    @test f16 == convolve!(p32, g64/fmax, f16)
    @test f16 ≈ fff/fmax

  end

end # @testset

#=

  for pre in [NFFT.LINEAR, NFFT.FULL, NFFT.TENSOR, NFFT.POLYNOMIAL]
@show pre
    p = plan_nfft(nodes, N, m = 5, σ = 2.0, precompute=pre)

    # ensure that convolve! works with mixed precisions
    for Tf in (Float16, Float64), Tg in (Float16, Float64)
        f = rand(Complex{Tf}, p.J)
        g = rand(Complex{Tg}, p.N)
        @test f == convolve!(p, g, f)
        @test g == convolve_transpose!(p, g, f)
    end
  end
=#

