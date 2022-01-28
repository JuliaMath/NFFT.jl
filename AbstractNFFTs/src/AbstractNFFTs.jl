module AbstractNFFTs

using Graphics: @mustimplement
using LinearAlgebra
using Printf

# interface
export AnyNFFTPlan, AbstractNFFTPlan, AbstractNNFFTPlan, 
       plan_nfft, mul!, size_in, size_out, nodes!

# optional
export apodization!, apodization_adjoint!, convolve!, convolve_adjoint!

# derived
export nfft, nfft_adjoint, ndft, ndft_adjoint

# misc
export TimingStats, accuracyParams, reltolToParams, paramsToReltol, PrecomputeFlags, LUT, FULL, FULL
   

include("misc.jl")
include("interface.jl")
include("derived.jl")


end