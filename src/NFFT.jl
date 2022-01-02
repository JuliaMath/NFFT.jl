module NFFT

using Printf
using Base.Cartesian
using FFTW
using Distributed
using SparseArrays
using LinearAlgebra
using Polyester
import Base.size
using Reexport

@reexport using AbstractNFFTs

export TimingStats
export NDFTPlan, NFFTPlan

export calculateToeplitzKernel, calculateToeplitzKernel!, convolveToeplitzKernel!

@enum PrecomputeFlags begin
  LUT = 1
  FULL = 2
  FULL_LUT = 3
end

#########################
# utility functions
#########################
include("utils.jl")
include("windowFunctions.jl")
include("precomputation.jl")

#################################
# currently implemented NFFTPlans
#################################

include("implementation.jl")
include("direct.jl")

#################
# factory methods
#################

"""
        plan_nfft(x::Matrix{T}, N::NTuple{D,Int}, rest...;  kargs...)

compute a plan for the NFFT of a size-`N` array at the nodes contained in `x`.
"""
function AbstractNFFTs.plan_nfft(::Type{Array}, x::Matrix{T}, N::NTuple{D,Int}, rest...;
                   timing::Union{Nothing,TimingStats} = nothing, kargs...) where {T,D}
  t = @elapsed begin
    p = NFFTPlan(x, N, rest...; kargs...)
  end
  if timing != nothing
    timing.pre = t
  end
  return p
end

"""
        plan_nfft(x::Matrix{T}, dim::Integer, N::NTuple{D,Int}, rest...;  kargs...)

compute a plan for the NFFT of a size-`N` array at the nodes contained in `x`
and along the direction `dim`.
"""
function AbstractNFFTs.plan_nfft(::Type{Array}, x::Matrix{T}, dim::Integer, N::NTuple{D,Int}, rest...; 
                   timing::Union{Nothing,TimingStats} = nothing, kargs...) where {T,D}
  t = @elapsed begin
    if size(x,1) != 1 && size(x,2) != 1
        throw(DimensionMismatch())
    end
    p = NFFTPlan(vec(x), dim, N, rest...; kargs...)
  end
  if timing != nothing
    timing.pre = t
  end
  return p
end



include("directional.jl")
include("multidimensional.jl")
include("samplingDensity.jl")
include("Toeplitz.jl")


function __init__()
  NFFT._use_threads[] = (Threads.nthreads() > 1)
end

end
