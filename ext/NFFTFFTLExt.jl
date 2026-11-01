# FFTLBackend: the FFT passes of NFFT plans with FFTL.jl. Its pass plans transform a strided view
# along one dimension in place, one thread per plan; NFFT.jl makes one plan per box and threads over
# the boxes, so no plan (and no buffer of one) is shared between tasks.
module NFFTFFTLExt

using NFFT, FFTL
using NFFT: FFTLBackend

NFFT.pass_plan(::FFTLBackend, ::Val{K}, v, d) where {K} = FFTL.plan_pass(v, d; sign=K)
NFFT.apply_pass!(p::FFTL.PassPlan, v) = (FFTL.apply!(p, v); nothing)

end
