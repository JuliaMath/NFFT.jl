module NFFTTools

export sdc
export calculateToeplitzKernel, calculateToeplitzKernel!, convolveToeplitzKernel!

using AbstractNFFTs: AbstractNFFTPlan, plan_nfft, nodes!
using AbstractNFFTs: convolve!, convolve_transpose!
using FFTW: fftshift, plan_fft, plan_ifft
using LinearAlgebra: adjoint, mul!
import FFTW # ESTIMATE (which is currently non-public)

include("samplingDensity.jl")
include("Toeplitz.jl")

end
