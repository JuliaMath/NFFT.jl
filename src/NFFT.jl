module NFFT

using Base.Cartesian
using FFTW
using Distributed
using SparseArrays
using LinearAlgebra
using CUDA
using Graphics: @mustimplement

export AbstractNFFTPlan, plan_nfft, nfft, nfft_adjoint, ndft, ndft_adjoint
import Base.size

@enum PrecomputeFlags begin
  LUT = 1
  FULL = 2
end

@enum Device begin
  CPU = 1
  CUDAGPU = 2
end

#########################
# abstract NFFT interface
#########################
include("Abstract.jl")

#################################
# currently implemented NFFTPlans
#################################
include("windowFunctions.jl")
include("precomputation.jl")

include("CpuNFFT.jl")
if CUDA.functional()
    using CUDA.CUSPARSE
    include("CuNFFT.jl")
end

#################
# factory methods
#################
"""
        plan_nfft(x::Matrix{T}, N::NTuple{D,Int}, rest...; device::Device=CPU, kargs...)

compute a plan for the NFFT of a size-`N` array at the nodes contained in `x`.

The computing device (CPU or GPU) can be set using the keyworkd argument `device` 
to NFFT.CPU or NFFT.CUDAGPU
"""
function plan_nfft(x::Matrix{T}, N::NTuple{D,Int}, rest...; device::Device=CPU, kargs...) where {T,D}
    if device==CPU
        p = NFFTPlan(x, N, rest...; kargs...)
    elseif device==CUDAGPU
        @assert CUDA.functional()==true "device $(dev) requires a functional CUDA setup"
        p = CuNFFTPlan(x, N, rest...; kargs...)
    else
        error("device $(dev) is not yet supported by NFFT.jl")
    end
    return p
end

"""
        plan_nfft(x::Vector{T}, dim::Integer, N::NTuple{D,Int}, rest...; device::Device=CPU, kargs...)

compute a plan for the NFFT of a size-`N` array at the nodes contained in `x`
and along the direction `dim`.
The computing device (CPU or GPU) can be set using the keyworkd argument `device`.
Currently only NFFT.CPU is supported.
"""
function plan_nfft(x::Vector{T}, dim::Integer, N::NTuple{D,Int}, rest...; device::Device=CPU, kargs...) where {T,D}
    if device==CPU
        p = NFFTPlan(x, dim, N, rest...; kargs...)
    elseif device==CUDAGPU
        error("device $(dev) doest not yet support directional NFFTs.")
    else
        error("device $(dev) is not yet supported by NFFT.jl")
    end
    return p
end

function plan_nfft(x::Matrix{T}, dim::Integer, N::NTuple{D,Int}, rest...; device::Device=CPU, kargs...) where {T,D}
    if size(x,1) != 1 && size(x,2) != 1
        throw(DimensionMismatch())
    end
    if device==CPU
        p = NFFTPlan(vec(x), dim, N, rest...; kargs...)
    elseif device==CUDAGPU
        error("device $(dev) doest not yet support directional NFFTs.")
    else
        error("device $(dev) is not yet supported by NFFT.jl")
    end
    return p
end

plan_nfft(x::AbstractMatrix{T}, N::NTuple{D,Int}, rest...; kwargs...) where {D,T} =
    plan_nfft(collect(x), N, rest...; kwargs...)

plan_nfft(x::AbstractVector, N::Integer, rest...; kwargs...) =
    plan_nfft(reshape(x,1,length(x)), (N,), rest...; kwargs...)

#########################
# high-level NFFT methods
#########################
"""
        nfft(x, f::AbstractArray{T,D}, rest...; device::Device=CPU, kwargs...)

calculates the NFFT of the array `f` for the nodes contained in the matrix `x`
The output is a vector of length M=`size(nodes,2)`
"""
function nfft(x, f::AbstractArray{T,D}, rest...; device::Device=CPU, kwargs...) where {T,D}
    p = plan_nfft(x, size(f), rest...; device=device, kwargs... )
    return nfft(p, f)
end

"""
        nfft_adjoint(x, N, fHat::AbstractArray{T,D}, rest...; device::Device=CPU, kwargs...)

calculates the adjoint NFFT of the vector `fHat` for the nodes contained in the matrix `x`.
The output is an array of size `N`
"""
function nfft_adjoint(x, N, fHat::AbstractVector{T}, rest...; device::Device=CPU, kwargs...) where T
    p = plan_nfft(x, N, rest...; device=device, kwargs...)
    return nfft_adjoint(p, fHat)
end

include("directional.jl")
include("multidimensional.jl")
include("samplingDensity.jl")
include("NDFT.jl")
include("Toeplitz.jl")

end
