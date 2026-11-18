using Metal

arrayTypes = [MtlArray]
floatTypes = [Float32] # Metal has no Float64

include(joinpath(@__DIR__, "..", "gpu.jl"))
