module NFFTTools

export sdc
export calculateToeplitzKernel, calculateToeplitzKernel!, convolveToeplitzKernel!

using AbstractNFFTs: AbstractNFFTPlan, plan_nfft, nodes!, size_in, size_out
using AbstractNFFTs: convolve!, convolve_transpose!
#using NFFT: NFFTPlan
using FFTW: fftshift, plan_fft, plan_ifft
using LinearAlgebra: adjoint, mul!
import FFTW # ESTIMATE

include("samplingDensity.jl")
include("Toeplitz.jl")

end
