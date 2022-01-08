module CuNFFT

using Reexport
@reexport using AbstractNFFTs
using NFFT
using LinearAlgebra
using AbstractFFTs
using CUDA

if CUDA.functional()
  using CUDA.CUSPARSE
  include("implementation.jl")
end



end
