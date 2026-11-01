# Runs the FFT backend tests with FFTL.jl, which is not registered, in a temporary environment:
#   julia test/fftl/run.jl [path to a local FFTL.jl]   (default: the main branch on GitHub)
using Pkg

root = joinpath(@__DIR__, "..", "..")
Pkg.activate(; temp=true)
Pkg.develop([PackageSpec(path=joinpath(root, "lib", "AbstractNFFTs")),
             PackageSpec(path=joinpath(root, "lib", "NFFTTools")),
             PackageSpec(path=root)])
Pkg.add(isempty(ARGS) ? PackageSpec(url="https://github.com/JuliaMath/FFTL.jl", rev="main") : PackageSpec(path=only(ARGS)))
Pkg.add(["Test", "Random", "LinearAlgebra"])

using Test, Random, LinearAlgebra, NFFT, FFTL
include(joinpath(root, "test", "contract.jl"))
include(joinpath(root, "test", "fftbackend.jl"))
fft_backend_tests(FFTLBackend())
