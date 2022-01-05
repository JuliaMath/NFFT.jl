module CuNFFT

using Reexport
@reexport using AbstractNFFTs
using LinearAlgebra
using CUDA

if CUDA.functional()
  using CUDA.CUSPARSE
  include("implementation.jl")
end



end