using Test
using NFFT
using Random

a = ones(ComplexF64,274,208)
b = ones(ComplexF64,274,208)
c = a[:] \ b[:]

Random.seed!(123)

include("test.jl")
include("performance.jl")
