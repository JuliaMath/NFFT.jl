module AbstractNFFTs

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
