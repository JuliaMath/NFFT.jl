module NFFT

using LinearAlgebra
using FFTW
using StaticArrays
using Reexport

@reexport using AbstractNFFTs

export NFFTBackend, NFFTPlan, NFCTPlan, NFSTPlan, NDFTPlan, NDCTPlan, NDSTPlan, NNDFTPlan
export AbstractWindow, KaiserBessel, ExpSemicircle, CustomWindow
export AbstractFFTBackend, FFTWBackend

"""
    NFFTBackend()

The AbstractNFFTs backend of NFFT.jl. Activated when the package is loaded; reactivate with
`NFFT.activate!()`.
"""
struct NFFTBackend <: AbstractNFFTBackend end
activate!() = AbstractNFFTs.set_active_backend!(NFFT)
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
include("real.jl")
include("direct.jl")

function __init__()
  activate!()
end

end
