using NFFT, LinearAlgebra#, CuNFFT

include("../Wrappers/NFFT3.jl")
include("../Wrappers/FINUFFT.jl")

ccall(("omp_set_num_threads",NFFT3.lib_path_nfft),Nothing,(Int64,),convert(Int64,Threads.nthreads()))
@info ccall(("nfft_get_num_threads",NFFT3.lib_path_nfft),Int64,())
NFFT._use_threads[] = (Threads.nthreads() > 1)

function nfft_performance_simple(;N = (1024,1024), J = prod(N), m = 4, 
  σ = 2.0, threading=false, pre=NFFT.LINEAR, T=Float64, blocking=true,
  storeDeconvolutionIdx=false, fftflags=NFFT.FFTW.MEASURE, ctor=NFFTPlan)

  timing = TimingStats()
  k = T.(rand(length(N),J) .- 0.5)
  k .= sortslices(k, dims=2) # sort nodes to gain cache locality
  
  fHat = randn(Complex{T}, J)
  f = randn(Complex{T}, N)

  #=if ctor == CuNFFT.CuNFFTPlan
    fHat = CuNFFT.CuArray(fHat)
    f = CuNFFT.CuArray(f)
  end=#

  NFFT._use_threads[] = threading

  tpre = @elapsed p = ctor(k, N; m, σ, window=:kaiser_bessel, precompute=pre, 
                            fftflags, storeDeconvolutionIdx, blocking)

  tadjoint = @elapsed mul!(f, adjoint(p), fHat; timing)
  ttrafo = @elapsed mul!(fHat, p, f; timing)

  @info tpre, ttrafo, tadjoint

  println(timing)

end
