module AbstractNFFTs

using Graphics: @mustimplement
using LinearAlgebra

# interface
export AbstractNFFTPlan, plan_nfft, size_in, size_out, nfft!, nfft_adjoint!

# derived
export nfft, nfft_adjoint, ndft, ndft_adjoint


include("interface.jl")
include("derived.jl")


end