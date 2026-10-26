using Test
using NFFT
using Random
using LinearAlgebra
using Zygote
using NFFTTools

# Contract tests of the AbstractNFFTs interface (implementation-independent), then NFFT.jl's own
# behaviour, then NFFTTools. Not run until migrated to the 1.0 interface: gpu.jl (GPU path),
# wrappers.jl (wrappers in Wrappers/).
include("contract.jl")

nfft_plan(k, N; kwargs...) = plan_nfft(NFFT.backend(), k, N; kwargs...)

@testset "NFFT" begin
  @testset "contract Float64" begin contract_tests(nfft_plan; T=Float64) end
  @testset "contract Float32" begin contract_tests(nfft_plan; T=Float32) end
  include("plan.jl")
  include("batch.jl")
  include("stages.jl")
  include("issues.jl")
  include("chainrules.jl")
  include("samplingDensity.jl")
  include("testToeplitz.jl")
end
