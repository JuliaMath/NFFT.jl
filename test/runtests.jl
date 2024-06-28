using Test
using NFFT
using Random
using LinearAlgebra
using FFTW
using NFFTTools
using Zygote
using JLArrays

Random.seed!(123)
areTypesDefined = @isdefined arrayTypes
arrayTypes = areTypesDefined ? arrayTypes : [JLArray]

@testset "NFFT" begin
  # If types were not defined we run everything
  if !areTypesDefined
    include("issues.jl")
    include("accuracy.jl")
    include("constructors.jl")
    include("performance.jl")
    include("testToeplitz.jl")
    include("samplingDensity.jl")
    include("gpu.jl")
    include("chainrules.jl")
    # Need to run after the other tests since they overload plan_*
    include("wrappers.jl")
  # If types were defined we only run GPU related tests
  else
    include("gpu.jl")
  end
end