using Test
using NFFT
using Random
using LinearAlgebra
using FFTW
using NFFTTools

Random.seed!(123)

include("accuracy.jl")
include("constructors.jl")
include("performance.jl")
include("testToeplitz.jl")
include("samplingDensity.jl")
include("cuda.jl")
# Need to run after the other tests since the overload plan_*
include("wrappers.jl")
