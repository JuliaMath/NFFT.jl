using NFFT, FFTW

FFTW.set_num_threads(Threads.nthreads())

# This is a reduced version of the actual problem
function _nfft!(p, f) 
  
  NFFT.apodization!(p, f, p.tmpVec)     # apodization uses Polyester-based threading or no threading depending on storeApodizationIdx

  @info "Timing After First FFT "       
  @time p.forwardFFT * p.tmpVec
  @info "Timing After Second FFT"
  @time p.forwardFFT * p.tmpVec

 return 
end

function doTimings()
  N = 1024;
  p1 = plan_nfft(zeros(2,N), (N,N); fftflags=NFFT.FFTW.MEASURE, storeApodizationIdx=false);
  f = zeros(ComplexF64,(N,N));
  
  # warmup FFT
  p1.forwardFFT * p1.tmpVec

  @info "\n\nWith Apodization Variant 1"
  _nfft!(p1, f);
 

  return
end

doTimings() # compiling
doTimings()
