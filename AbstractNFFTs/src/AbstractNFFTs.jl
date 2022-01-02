module AbstractNFFTs

using Graphics: @mustimplement
using LinearAlgebra

# interface
export AbstractNFFTPlan, plan_nfft, numFourierSamples

# derived
export nfft, nfft_adjoint, ndft, ndft_adjoint, nfft!, nfft_adjoint!

import Base.size

include("interface.jl")
include("derived.jl")


end