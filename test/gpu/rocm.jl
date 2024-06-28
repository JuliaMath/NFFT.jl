using AMDGPU

arrayTypes = [ROCArray]

include(joinpath(@__DIR__(), "..", "runtests.jl"))