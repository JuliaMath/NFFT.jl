module AbstractNFFTs

using Graphics: @mustimplement
using LinearAlgebra
using Printf

# interface
export AnyNFFTPlan, AbstractNFFTPlan, AbstractNFCTPlan, AbstractNNFFTPlan, 
       plan_nfft, plan_nfct, mul!, size_in, size_out, nodes!

# optional
export apodization!, apodization_adjoint!, convolve!, convolve_adjoint!

# derived
export nfft, nfft_adjoint, ndft, ndft_adjoint, nfct, nfct_transposed, ndct, ndct_transposed

# misc
export TimingStats, accuracyParams, reltolToParams, paramsToReltol, 
       PrecomputeFlags, LINEAR, FULL, TENSOR, POLYNOMIAL
   

include("misc.jl")
include("interface.jl")
include("derived.jl")


end
