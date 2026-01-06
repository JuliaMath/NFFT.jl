module NFFTGPUArraysExt

using NFFT, NFFT.AbstractNFFTs
using NFFT.SparseArrays, NFFT.LinearAlgebra, NFFT.FFTW
using GPUArrays, GPUArrays.KernelAbstractions, Adapt
using GPUArrays.KernelAbstractions.Extras: @unroll

include("implementation.jl")
include("precomputation.jl")

end
