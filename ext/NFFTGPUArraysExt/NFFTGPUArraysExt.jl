module NFFTGPUArraysExt

using NFFT, NFFT.AbstractNFFTs
using NFFT.SparseArrays, NFFT.LinearAlgebra, NFFT.FFTW
using GPUArrays, Adapt

include("implementation.jl")

end
