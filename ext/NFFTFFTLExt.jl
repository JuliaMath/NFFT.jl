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

# the real transforms of the RNFFT as native r2c/c2r passes instead of the complex fallback of the
# same length. FFTL has the passes only from the passes layer (branch, not on main yet): against an
# older FFTL the complex fallback stays. The passes must support the RNFFT's layout — the real grid
# boxes are interior views of the padded grid, so the c2r output is not dense for D ≥ 2 (the
# realpass of the current branches assumes a dense output and fails there; the cleanup task
# notes/fftl-task-cleanup.md fixes it). D = 1 is dense and runs.
if isdefined(FFTL, :plan_r2c_pass)
  NFFT.rfft_plan(::FFTLBackend, x, X) =
    FFTL.plan_r2c_pass(x, 1; threads=get(task_local_storage(), :nfft_fftl_threads, 1))
  NFFT.brfft_plan(::FFTLBackend, X, x) =
    FFTL.plan_c2r_pass(X, 1; n=size(x, 1), threads=get(task_local_storage(), :nfft_fftl_threads, 1))
end

end
