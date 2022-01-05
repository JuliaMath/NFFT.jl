using Test
using NFFT
using Random
#using CuNFFT

Random.seed!(123)

include("constructors.jl")
include("accuracy.jl")
include("performance.jl")
include("testToeplitz.jl")
include("samplingDensity.jl")
