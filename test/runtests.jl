using Test
using NFFT
using Random
using LinearAlgebra

# Contract tests of the AbstractNFFTs interface (implementation-independent). The other test files
# of NFFT.jl 0.14 target its old interface and are migrated step by step.
include("contract.jl")

nfft_plan(k, N; kwargs...) = plan_nfft(NFFT.backend(), k, N; kwargs...)

@testset "NFFT" begin
  @testset "contract Float64" begin contract_tests(nfft_plan; T=Float64) end
  @testset "contract Float32" begin contract_tests(nfft_plan; T=Float32) end
end
