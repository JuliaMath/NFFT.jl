# NFFT.jl 1.x with FFTL as FFT backend: the ablation against the FFTW backend (`nfft.jl`, which this
# includes and of which it overrides only the `fft` keyword of the plans). The plans of the real
# transforms go through complex FFTs of the same length (FFTL has no real-to-complex passes yet).
# The first plan of a length compiles its kernels, which lands in `t_plan`.
include(joinpath(@__DIR__, "nfft.jl"))
import FFTL

fft_kw() = (; fft = NFFT.FFTLBackend())
