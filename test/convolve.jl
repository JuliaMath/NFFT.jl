#=
Test convolution (interpolation) and its adjoint in isolation

One must be careful with Float16 output arrays here,
because plan.windowLinInterp has values above 1e8
that exceed the 6e4 maximum of Float16.

FFTW does not support Float16 plans as of 2026-01-01.
=#

using JLArrays: JLArray
using NFFT: plan_nfft
using AbstractNFFTs: convolve!, convolve_transpose!
import NFFT # LINEAR, FULL, TENSOR, POLYNOMIAL
using Test: @test, @testset, @test_throws


# Determine worst precision of two types
# (Does Julia already have such a method?)
function worst(T1::Type{<:Number}, T2::Type{<:Number})
    return promote_type(T1, T2) == T1 ? T2 : T1
end
worst(a, b, c) = worst(worst(a, b), c)

@testset "worst" begin
  @test worst(Float64, Float16, Float32) == Float16
  @test worst(Complex{Float16}, Complex{BigFloat}) == Complex{Float16}
end


N = (9, 8) # grid
Ñ = 2 .* N # for g
J = 101 # nodes for "fHat"
nodes = rand(BigFloat, length(N), J) .- 0.5

F16 = Float16
F32 = Float32
F64 = Float64
C16 = Complex{F16}
C32 = Complex{F32}


@testset "convolve-throws" begin
  # throw error when input is complex and output is real
  p = plan_nfft(Float32.(nodes), N, m = 5, σ = 2.0, precompute=NFFT.LINEAR)
  @test_throws ArgumentError convolve!(p, ones(C32, Ñ), zeros(Float32, J))
  @test_throws ArgumentError convolve_transpose!(p, ones(C32, J), zeros(Float32,  Ñ))

  # or when dimensions mismatch
  @test_throws DimensionMismatch convolve!(p, ones(C32, Ñ), zeros(C32, 2))
  @test_throws DimensionMismatch convolve!(p, ones(C32, N), zeros(C32, J))
  @test_throws DimensionMismatch convolve_transpose!(p, ones(C32, 2), zeros(C32, Ñ))
  @test_throws DimensionMismatch convolve_transpose!(p, ones(C32, J), zeros(C32, N))
end


@testset "convolve!" begin

  for pre in [NFFT.LINEAR, NFFT.FULL, NFFT.TENSOR, NFFT.POLYNOMIAL]
   for arr in (Array, JLArray)

    p32 = plan_nfft(arr, Float32.(nodes), N, m = 5, σ = 2.0, precompute=pre)
    p64 = plan_nfft(arr, Float64.(nodes), N, m = 5, σ = 2.0, precompute=pre)
    @assert Ñ == p64.Ñ

    _similar = (T, dims) -> similar(p64.tmpVec, T, dims)

    for tfun in (identity, t -> Complex{t}) # test both Real and Complex
      T16 = tfun(F16)
      T32 = tfun(F32)
      T64 = tfun(F64)

      g64 = _similar(T64, Ñ)
      g64 .= rand(T64, Ñ)
      fff = _similar(T64, J) # ground-truth reference
      @test fff === convolve!(p64, g64, fff)

      fmax = maximum(abs, fff) # need to avoid avoid overflow with Float16

      for Tg in (T16, T32, T64), Tf in (T32, T64)
        f = _similar(Tf, J)
        @test f === convolve!(p64, Tg.(g64), f)
        T = worst(Tg, Tf)
        @test f/fmax ≈ T.(fff/fmax)

        @test f === convolve!(p32, Tg.(g64), f)
        T = worst(Tg, Tf, T32)
        @test f / fmax ≈ T.(fff / fmax)
      end

      # Float16 test with careful scaling:
      f16 = _similar(T16, J)
      @test f16 === convolve!(p64, g64/fmax, f16)
      @test f16 ≈ fff / fmax
      @test f16 === convolve!(p32, g64/fmax, f16)
      @test f16 ≈ fff / fmax
    end # tfun
   end # arr
  end # pre

end # @testset


@testset "convolve_transpose!" begin

  for pre in [NFFT.LINEAR, NFFT.FULL, NFFT.TENSOR, NFFT.POLYNOMIAL]
   for arr in (Array, JLArray)
    p32 = plan_nfft(Float32.(nodes), N, m = 5, σ = 2.0, precompute=pre)
    p64 = plan_nfft(Float64.(nodes), N, m = 5, σ = 2.0, precompute=pre)
    @assert Ñ == p64.Ñ

    _similar = (T, dims) -> similar(p64.tmpVec, T, dims)

    for tfun in (identity, t -> Complex{t}) # test both Real and Complex
      T16 = tfun(F16)
      T32 = tfun(F32)
      T64 = tfun(F64)

      f64 = _similar(T64, J)
      f64 .= rand(T64, J)
      ggg = _similar(T64, Ñ) # ground-truth reference
      @test ggg === convolve_transpose!(p64, f64, ggg)

      gmax = maximum(abs, ggg) # need to avoid avoid overflow with Float16

      for Tg in (T32, T64), Tf in (T16, T32, T64)
        g = similar(ggg, Tg)
        @test g === convolve_transpose!(p64, Tf.(f64), g)
        T = worst(Tg, Tf)
        @test g / gmax ≈ T.(ggg / gmax)

        @test g === convolve_transpose!(p32, Tf.(f64), g)
        T = worst(Tg, Tf, T32)
        @test g / gmax ≈ T.(ggg / gmax)
      end

      # Float16 test with careful scaling:
      g16 = _similar(T16, Ñ)
      @test g16 === convolve_transpose!(p64, f64/gmax, g16)
      @test g16 ≈ ggg / gmax
      @test g16 === convolve_transpose!(p32, f64/gmax, g16)
      @test g16 ≈ ggg / gmax
    end # tfun
   end # arr
  end # pre
end # @testset
