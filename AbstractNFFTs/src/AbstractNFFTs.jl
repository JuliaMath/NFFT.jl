module AbstractNFFTs

using Graphics: @mustimplement
using LinearAlgebra
using Printf

# interface
export AbstractNFFTPlan, plan_nfft, size_in, size_out, nfft!, nfft_adjoint!, nodes!

# optional
export apodization!, apodization_adjoint!, convolve!, convolve_adjoint!

# derived
export nfft, nfft_adjoint, ndft, ndft_adjoint

# misc
export TimingStats, PrecomputeFlags, LUT, FULL, FULL
   

include("misc.jl")
include("interface.jl")
include("derived.jl")


end