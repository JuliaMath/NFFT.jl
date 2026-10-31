using Test
using NFFT
using Random
using LinearAlgebra
using Zygote
using NFFTTools
using SparseArrays
using FFTA

# Contract tests of the AbstractNFFTs interface (implementation-independent), then NFFT.jl's own
# behaviour, then NFFTTools, the GPU code on the CPU and Aqua. The tests on GPU devices run
# separately (test/gpu/run.jl, on Buildkite).
include("contract.jl")

nfft_plan(k, N; kwargs...) = plan_nfft(NFFT.backend(), k, N; kwargs...)

@testset "NFFT" begin
  @testset "contract Float64" begin contract_tests(nfft_plan; T=Float64) end
  @testset "contract Float32" begin contract_tests(nfft_plan; T=Float32) end
  include("interface.jl")
  include("plan.jl")
  include("batch.jl")
  include("real.jl")
  include("rnfft.jl")
  include("nnfft.jl")
  include("highdim.jl")
  include("stages.jl")
  include("ablation.jl")
  include("stability.jl")
  include("fftbackend.jl")
  include("precision.jl")
  include("issues.jl")
  include("chainrules.jl")
  include("enzyme.jl")
  include("samplingDensity.jl")
  include("testToeplitz.jl")
end

# the GPU code on the CPU (JLArrays); the GPU devices run in test/gpu/run.jl
include("gpu/jlarrays.jl")
include("aqua.jl")
