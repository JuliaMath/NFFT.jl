module NFFTTools

export sdc, sdc!
export calculateToeplitzKernel, calculateToeplitzKernel!, convolveToeplitzKernel!

using AbstractNFFTs: AbstractNFFTPlan, plan_nfft, nodes, nodes!, size_in, size_out, accuracy
using AbstractNFFTs: oversampled_size, resampling!, resampling_adjoint!
using FFTL: fftshift, plan_fft, plan_ifft
using LinearAlgebra: adjoint, mul!

include("samplingDensity.jl")
include("Toeplitz.jl")

end
