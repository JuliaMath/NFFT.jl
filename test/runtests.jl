using Test
using NFFT
using Random
using LinearAlgebra
using FFTW
using NFFTTools
using Zygote

Random.seed!(123)

include("issues.jl")
include("accuracy.jl")
include("constructors.jl")
include("performance.jl")
include("testToeplitz.jl")
include("samplingDensity.jl")
include("cuda.jl")
include("chainrules.jl")
# Need to run after the other tests since they overload plan_*
include("wrappers.jl")
