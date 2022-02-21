using Test
using NFFT
using Random
using LinearAlgebra
using FFTW
using NFFTTools

Random.seed!(123)

include("accuracy.jl")
include("wrappers.jl")
include("constructors.jl")
include("performance.jl")
include("testToeplitz.jl")
include("samplingDensity.jl")
include("cuda.jl")
