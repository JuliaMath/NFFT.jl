# FFTABackend: the FFT passes of NFFT plans with FFTA.jl.
module NFFTFFTAExt

using NFFT, FFTA, LinearAlgebra
using NFFT: FFTABackend, FFTW
using NFFT.FFTW: AbstractFFTs

# FFTA's plans work out of place: each plan keeps a buffer of its box.
struct FFTAPass{P,B}
  plan::P
  buf::B
end

# With FFTW loaded, AbstractFFTs.plan_fft of a strided array dispatches to FFTW; `invoke` selects
# FFTA's method.
function NFFT.pass_plan(::FFTABackend, ::Val{K}, v::AbstractArray{C,D}, d) where {K,C,D}
  plan = K == -1 ? invoke(AbstractFFTs.plan_fft, Tuple{AbstractArray{C,D},Any}, v, d) :
                   invoke(AbstractFFTs.plan_bfft, Tuple{AbstractArray{C,D},Any}, v, d)
  FFTAPass(plan, Array{C}(undef, size(v)))
end

NFFT.apply_pass!(p::FFTAPass, v) = (mul!(p.buf, p.plan, v); copyto!(v, p.buf); nothing)

end
