using NFFT

N = 1024 
M = N*N 
m = 4 
LUTSize=20000
σ = 2.0
pre=NFFT.LUT 
T=Float64 
storeApodizationIdx=false
fftflags=NFFT.FFTW.ESTIMATE

x = T.(rand(2,M) .- 0.5)
  
@time p = NFFTPlan(x, (N,N); m, σ, window=:kaiser_bessel, 
           LUTSize, precompute=pre, fftflags, storeApodizationIdx)
