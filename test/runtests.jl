using Test
using NFFT
using Random
using LinearAlgebra
using FFTW
#using CuNFFT
#using NFFTTools

Random.seed!(123)


include("wrappers.jl")
include("constructors.jl")
include("accuracy.jl")
include("performance.jl")
#include("testToeplitz.jl")
#include("samplingDensity.jl")
