# Runs the GPU tests in a temporary environment: julia test/gpu/run.jl cuda|rocm|metal
using Pkg

const BACKENDS = Dict("cuda" => "CUDA", "rocm" => "AMDGPU", "metal" => "Metal")
backend = only(ARGS)
haskey(BACKENDS, backend) || error("unknown backend $backend, expected one of $(keys(BACKENDS))")

root = joinpath(@__DIR__, "..", "..")
Pkg.activate(; temp=true)
Pkg.develop([PackageSpec(path=joinpath(root, "AbstractNFFTs")),
             PackageSpec(path=joinpath(root, "NFFTTools")),
             PackageSpec(path=root)])
Pkg.add([BACKENDS[backend], "Test", "Random", "LinearAlgebra"])

include(joinpath(@__DIR__, "$backend.jl"))
