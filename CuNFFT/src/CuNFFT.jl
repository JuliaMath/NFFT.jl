module CuNFFT

using Reexport
@reexport using AbstractNFFTs
using NFFT
using LinearAlgebra
using AbstractFFTs
using CUDA
using CUDA.CUSPARSE

function __init__()
  @warn """Please be aware that CuNFFT.jl has been deprecated by a package extension in NFFT.jl.
  As a result, no additional packages are required to run an NFFT on a GPU.

  To remove this warning when continuing to use CuNFFT.jl, please pin CuNFFT.jl to version 0.3.7."""
end

include("implementation.jl")

end
