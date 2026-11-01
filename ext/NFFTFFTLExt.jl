# FFTLBackend: the FFT passes of NFFT plans with FFTL.jl. Its pass plans transform a strided view
# along one dimension in place. NFFT.jl makes one plan per box and threads over the boxes, so no
# plan (and no buffer of one) is shared between tasks; a pass of a single box (1D) is planned with
# the thread count of `planning`, which FFTL uses for its four-step transform. Such a plan owns
# per-thread buffers and is applied by one task at a time.
module NFFTFFTLExt

using NFFT, FFTL
using NFFT: FFTLBackend

# task-local, so that plans made concurrently on other tasks keep their own count
NFFT.planning(f, ::FFTLBackend, nthreads) = task_local_storage(f, :nfft_fftl_threads, nthreads)
NFFT.pass_plan(::FFTLBackend, ::Val{K}, v, d) where {K} =
  FFTL.plan_pass(v, d; sign=K, threads=get(task_local_storage(), :nfft_fftl_threads, 1))
NFFT.apply_pass!(p::FFTL.PassPlan, v) = (FFTL.apply!(p, v); nothing)

end
