using Test
using NFFT
using Random

Random.seed!(123)

include("test.jl")
include("performance.jl")
