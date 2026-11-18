# The FFTW backend: the FFT passes of NFFT plans with FFTW on the CPU. FFTW is a weak dependency:
# the backend is an option, `plan_nfft(...; fft = FFTWBackend())` needs `using FFTW`. The per-dimension
# passes are single-threaded FFTW plans threaded by NFFT.jl (FFTW's own threading with ESTIMATE
# produces plans up to 17× slower than MEASURE); a single 1D pass is threaded by FFTW itself.
module NFFTFFTWExt

using NFFT, FFTW
using NFFT: FFTWBackend
using LinearAlgebra

NFFT.pass_plan(fft::FFTWBackend, ::Val{K}, v, d) where {K} =
  K == FFTW.FORWARD ? FFTW.plan_fft!(v, d; flags=fft.flags) : FFTW.plan_bfft!(v, d; flags=fft.flags)

# The thread count of FFTW's planner is global; planning restores it.
function NFFT.planning(f, ::FFTWBackend, n)
  old = FFTW.get_num_threads()
  FFTW.set_num_threads(n)
  try
    f()
  finally
    FFTW.set_num_threads(old)
  end
end

# Real FFTs along dimension 1: native r2c to the half spectrum and its inverse c2r (unnormalised).
NFFT.rfft_plan(fft::FFTWBackend, x, X) =
  FFTW.rFFTWPlan{eltype(x),FFTW.FORWARD,false,ndims(x)}(x, X, (1,), fft.flags, FFTW.NO_TIMELIMIT)
NFFT.brfft_plan(fft::FFTWBackend, X, x) =
  FFTW.rFFTWPlan{eltype(X),FFTW.BACKWARD,false,ndims(x)}(X, x, (1,), fft.flags, FFTW.NO_TIMELIMIT)

# The full-transform shortcuts of the RNFFT stage interface: FFTW's multidimensional real transforms.
NFFT.full_brfft(fft::FFTWBackend, X, n) = FFTW.brfft(X, n)
NFFT.full_rfft(fft::FFTWBackend, g, n) = FFTW.rfft(g)

end
