module NFFT

using Printf
using Base.Cartesian
using FFTW
using Distributed
using SparseArrays
using LinearAlgebra
using FLoops
import Base.size
using Reexport
using SpecialFunctions: besseli, besselj
using BasicInterpolators

#using StaticArrays

@reexport using AbstractNFFTs

export NFFTBackend, NDFTPlan, NDCTPlan, NDSTPlan, NNDFTPlan, 
       NFFTPlan, NFFTParams


#########################
# utility functions
#########################
include("utils.jl")
include("windowFunctions.jl")

#################################
# implementations
#################################

include("implementation.jl")
include("direct.jl")
include("precomputation.jl")

#################
# factory methods
#################
struct NFFTBackend <: AbstractNFFTBackend end
activate!() = AbstractNFFTs.set_active_backend!(NFFT)
backend() = NFFTBackend()

"""
    NFFT.plan_nfft(k::Matrix{T}, N::NTuple{D,Int}, rest...;  kargs...)

compute a plan for the NFFT of a size-`N` array at the nodes contained in `k`.
"""
function AbstractNFFTs.plan_nfft(::NFFTBackend, ::Type{<:Array}, k::Matrix{T}, N::NTuple{D,Int}, rest...;
                   timing::Union{Nothing,TimingStats} = nothing, kargs...) where {T,D}
  t = @elapsed begin
    p = NFFTPlan(k, N, rest...; kargs...)
  end
  if timing != nothing
    timing.pre = t
  end
  return p
end

include("directional.jl")
include("deconvolution.jl")
include("convolution.jl")

function __init__()
  NFFT._use_threads[] = (Threads.nthreads() > 1)
  activate!()
end

include("precompile.jl")

end
