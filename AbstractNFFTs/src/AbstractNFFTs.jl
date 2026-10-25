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
       plan_nfft, plan_nfct, plan_nfst, plan_nnfft, mul!, size_in, size_out, nodes!, nodes,
       accuracy

# optional stage interface (terms of Knopp et al. 2023)
export oversampled_size, resampling_correction!, oversampled_fft!, resampling!,
       resampling_adjoint!, oversampled_bfft!, resampling_correction_adjoint!

# derived
export nfft, nfft_adjoint, ndft, ndft_adjoint,
       nfct, nfct_transpose, ndct, ndct_transpose,
       nfst, nfst_transpose

# options shared by all implementations
export DEFAULT_RELTOL, GenericOptions, generic_options, check_options

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
