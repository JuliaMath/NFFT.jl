module NFFTTools

export sdc
export calculateToeplitzKernel, calculateToeplitzKernel!, convolveToeplitzKernel!

include("samplingDensity.jl")
include("Toeplitz.jl")

end
