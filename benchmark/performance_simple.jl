using NFFT, LinearAlgebra, CuNFFT

include("../Wrappers/NFFT3.jl")
include("../Wrappers/FINUFFT.jl")

NFFT.FFTW.set_num_threads(Threads.nthreads())
ccall(("omp_set_num_threads",NFFT3.lib_path_nfft),Nothing,(Int64,),convert(Int64,Threads.nthreads()))
@info ccall(("nfft_get_num_threads",NFFT3.lib_path_nfft),Int64,())
NFFT._use_threads[] = (Threads.nthreads() > 1)

function nfft_performance_simple(;N = (1024,1024), M = prod(N), m = 4, 
  σ = 2.0, threading=false, pre=NFFT.LINEAR, T=Float64, blocking=true,
  storeApodizationIdx=false, fftflags=NFFT.FFTW.MEASURE, ctor=NFFTPlan)

  timing = TimingStats()
  x = T.(rand(length(N),M) .- 0.5)
  x .= sortslices(x, dims=2) # sort nodes to gain cache locality
  
  fHat = randn(Complex{T}, M)
  f = randn(Complex{T}, N)
s
  #=if ctor == CuNFFT.CuNFFTPlan
    fHat = CuNFFT.CuArray(fHat)
    f = CuNFFT.CuArray(f)
  end=#

  NFFT._use_threads[] = threading
  NFFT.FFTW.set_num_threads( threading ? Threads.nthreads() : 1)

  tpre = @elapsed p = ctor(x, N; m, σ, window=:kaiser_bessel, precompute=pre, 
                            fftflags, storeApodizationIdx, blocking)

  tadjoint = @elapsed CUDA.@sync mul!(f, adjoint(p), fHat; timing)
  ttrafo = @elapsed CUDA.@sync mul!(fHat, p, f; timing)

  @info tpre, ttrafo, tadjoint

  println(timing)

end
