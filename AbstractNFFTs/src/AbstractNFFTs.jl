module AbstractNFFTs

using LinearAlgebra
using Printf

# Remove this difference once 1.11 or higher becomes lower bound
if VERSION >= v"1.11"
  using Base.ScopedValues
else
  using ScopedValues
end


# interface
export AbstractNFFTBackend, nfft_backend, with
export AbstractFTPlan, AbstractRealFTPlan, AbstractComplexFTPlan,
       AbstractNFFTPlan, AbstractNFCTPlan, AbstractNFSTPlan, AbstractNNFFTPlan, 
       plan_nfft, plan_nfct, plan_nfst, mul!, size_in, size_out, nodes!

# optional
export deconvolve!, deconvolve_transpose!, convolve!, convolve_transpose!

# derived
export nfft, nfft_transpose, nfft_adjoint, ndft, ndft_adjoint, 
       nfct, nfct_transpose, nftct_adjoint, ndct, ndct_transpose,
       nfst, nfst_transpose, nfst_adjoint

# misc
export TimingStats, accuracyParams, reltolToParams, paramsToReltol, 
       PrecomputeFlags, LINEAR, FULL, TENSOR, POLYNOMIAL
   

include("misc.jl")
include("interface.jl")
include("derived.jl")


@static if !isdefined(Base, :get_extension)
  import Requires
end
   
@static if !isdefined(Base, :get_extension)
  function __init__()
    Requires.@require ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4" begin
      include("../ext/AbstractNFFTsChainRulesCoreExt.jl")
    end
  end
end

end
