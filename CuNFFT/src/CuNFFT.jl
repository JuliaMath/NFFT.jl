module CuNFFT

using Reexport
@reexport using AbstractNFFTs
using LinearAlgebra
using CUDA

import Base.size

if CUDA.functional()
  using CUDA.CUSPARSE
  include("implementation.jl")
end



end