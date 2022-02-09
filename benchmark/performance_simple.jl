using NFFT, LinearAlgebra

include("../Wrappers/NFFT3.jl")
include("../Wrappers/FINUFFT.jl")

NFFT.FFTW.set_num_threads(Threads.nthreads())
ccall(("omp_set_num_threads",NFFT3.lib_path_nfft),Nothing,(Int64,),convert(Int64,Threads.nthreads()))
@info ccall(("nfft_get_num_threads",NFFT3.lib_path_nfft),Int64,())
NFFT._use_threads[] = (Threads.nthreads() > 1)

function nfft_performance_simple(;N = 1024, M = N*N, m = 4, 
  σ = 2.0, threading=false, pre=NFFT.LUT, T=Float64, blocking=true,
  storeApodizationIdx=false, fftflags=NFFT.FFTW.MEASURE, ctor=NFFTPlan)

  timing = TimingStats()
  x = T.(rand(2,M) .- 0.5)
  x .= sortslices(x, dims=2) # sort nodes to gain cache locality
  
  fHat = randn(Complex{T}, M)
  f = randn(Complex{T}, (N,N))

  NFFT._use_threads[] = threading
  NFFT.FFTW.set_num_threads( threading ? Threads.nthreads() : 1)

  tpre = @elapsed p = ctor(x, (N,N); m, σ, window=:kaiser_bessel, precompute=pre, 
                            fftflags, storeApodizationIdx, blocking)

  mul!(f, adjoint(p), fHat; timing)
  tadjoint = @elapsed mul!(f, adjoint(p), fHat; timing)
  mul!(fHat, p, f; timing)
  ttrafo = @elapsed mul!(fHat, p, f; timing)

  @info tpre, ttrafo, tadjoint

  println(timing)

end