module AbstractNFFTs

using Graphics: @mustimplement
using LinearAlgebra
using Printf

# interface
export AbstractFTPlan, AbstractRealFTPlan, AbstractComplexFTPlan,
       AbstractNFFTPlan, AbstractNFCTPlan, AbstractNFSTPlan, AbstractNNFFTPlan, 
       plan_nfft, plan_nfct, plan_nfst, mul!, size_in, size_out, nodes!

# optional
export deconvolve!, deconvolve_transpose!, convolve!, convolve_transpose!

# derived
export nfft, nfft_adjoint, ndft, ndft_adjoint, 
       nfct, nfct_transpose, ndct, ndct_transpose,
       nfst, nfst_transpose

# misc
export TimingStats, accuracyParams, reltolToParams, paramsToReltol, 
       PrecomputeFlags, LINEAR, FULL, TENSOR, POLYNOMIAL
   

include("misc.jl")
include("interface.jl")
include("derived.jl")


end
