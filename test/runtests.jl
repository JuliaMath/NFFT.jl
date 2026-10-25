using Test
using NFFT
using Random
using LinearAlgebra
using Zygote

# Contract tests of the AbstractNFFTs interface (implementation-independent), then NFFT.jl's own
# behaviour. Not run until migrated to the 1.0 interface: samplingDensity.jl and testToeplitz.jl
# (need NFFTTools), gpu.jl (GPU path), wrappers.jl (wrappers in Wrappers/).
include("contract.jl")

nfft_plan(k, N; kwargs...) = plan_nfft(NFFT.backend(), k, N; kwargs...)

@testset "NFFT" begin
  @testset "contract Float64" begin contract_tests(nfft_plan; T=Float64) end
  @testset "contract Float32" begin contract_tests(nfft_plan; T=Float32) end
  include("plan.jl")
  include("stages.jl")
  include("issues.jl")
  include("chainrules.jl")
end
