using CUDA

arrayTypes = [CuArray]

include(joinpath(@__DIR__(), "..", "runtests.jl"))