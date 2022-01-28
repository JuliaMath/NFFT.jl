using NFFT, LinearAlgebra

function nfft_performance_simple(;N = 64, M = N*N, m = 5, LUTSize=100000,
  σ = 2.0, threading=false, pre=NFFT.LUT, T=Float64, 
  storeApodizationIdx=true, fftflags=NFFT.FFTW.MEASURE)

  timing = TimingStats()
  x = T.(rand(2,M) .- 0.5)
  fHat = Complex{T}.(rand(M)*1im)
  NFFT._use_threads[] = threading
  NFFT.FFTW.set_num_threads( threading ? Threads.nthreads() : 1)

  tpre = @elapsed p = plan_nfft(x, (N,N); m, σ, window=:kaiser_bessel, LUTSize, precompute=pre, timing, fftflags, storeApodizationIdx)
  f = similar(fHat, p.N)
  tadjoint = @elapsed fApprox = mul!(f, adjoint(p), fHat; timing)
  ttrafo = @elapsed mul!(fHat, p, fApprox; timing)

  @info tpre, ttrafo, tadjoint

  println(timing)

end