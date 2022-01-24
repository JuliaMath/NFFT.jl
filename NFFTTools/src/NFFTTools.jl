module NFFTTools


using LinearAlgebra
using AbstractFFTs, FFTW  # FFTW just because of FFTW.ESTIMATE
using AbstractNFFTs

export sdc
export calculateToeplitzKernel, calculateToeplitzKernel!, convolveToeplitzKernel!

include("samplingDensity.jl")
include("Toeplitz.jl")

end
