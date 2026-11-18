# NFFT.jl 1.x with FFTL as FFT backend, made explicit (the default; `nfft.jl`, which this includes
# and of which it overrides only the `fft` keyword of the plans). For ad-hoc backend comparisons;
# the standard suites run `NFFT` with the default backend alone. The first plan of a length
# compiles its kernels, which lands in `t_plan`.
include(joinpath(@__DIR__, "nfft.jl"))
import FFTL

fft_kw() = (; fft = NFFT.FFTLBackend())
