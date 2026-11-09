using CUDA

arrayTypes = [CuArray]
floatTypes = [Float32, Float64]

include(joinpath(@__DIR__, "..", "gpu.jl"))
