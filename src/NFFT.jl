module NFFT

using LinearAlgebra
using FFTW
using StaticArrays
using Reexport

@reexport using AbstractNFFTs

export NFFTBackend, NFFTPlan, NFCTPlan, NFSTPlan, RNFFTPlan, NNFFTPlan, NDFTPlan, NDCTPlan, NDSTPlan, NNDFTPlan
export AbstractWindow, KaiserBessel, ExpSemicircle, CustomWindow
export AbstractFFTBackend, FFTWBackend, FFTABackend

"""
    NFFTBackend()

The AbstractNFFTs backend of NFFT.jl. Activated when the package is loaded; reactivate with
`NFFT.activate!()`.
"""
struct NFFTBackend <: AbstractNFFTBackend end
activate!() = AbstractNFFTs.set_active_backend!(NFFTBackend())
backend() = NFFTBackend()

include("util.jl")
include("windows.jl")
include("kernel.jl")
include("geometry.jl")
include("binning.jl")
include("fft.jl")
include("resampling.jl")
include("correction.jl")
include("parameters.jl")
include("batch.jl")
include("plan.jl")
include("ablation.jl")
include("real.jl")
include("rnfft.jl")
include("plans.jl")
include("nnfft.jl")
include("direct.jl")
include("precompile.jl")

function __init__()
  activate!()
end

end
