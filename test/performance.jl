using NFFT
using FFTW
import NFFT3
using DataFrames

FFTW.set_num_threads(Threads.nthreads())

### performance test ###
function nfft_performance_1()
  println("\n\n ##### nfft_performance_1 - simple ##### \n\n")

  timing = TimingStats()

  let m = 3, σ = 2.0
    @info "NFFT Performance Test 1D"
    let N = 2^19, M = N, x = rand(M) .- 0.5, fHat = rand(M)*1im
      for pre in [NFFT.LUT, NFFT.FULL]
        @info "* precomputation = $pre"
        p = plan_nfft(x, N; m, σ, precompute=pre, timing)
        fApprox = nfft_adjoint(p, fHat; timing)
        nfft(p, fApprox; timing)

        println(timing)
      end
    end

    @info "NFFT Performance Test 2D"
    let N = 1024, M = N*N, x2 = rand(2,M) .- 0.5, fHat = rand(M)*1im
      for pre in [NFFT.LUT, NFFT.FULL]
        @info "* precomputation = $pre"
        p = plan_nfft(x2, (N,N); m, σ, precompute=pre, timing)
        fApprox = nfft_adjoint(p, fHat; timing)
        nfft(p, fApprox; timing)

        println(timing)
      end
    end

    @info "NFFT Performance Test 3D"
    let N = 32, M = N*N*N, x3 = rand(3,M) .- 0.5, fHat = rand(M)*1im
      for pre in [NFFT.LUT, NFFT.FULL]
        @info "* precomputation = $pre"
        p = plan_nfft(x3, (N,N,N); m, σ, precompute=pre, timing)
        fApprox = nfft_adjoint(p, fHat; timing)
        nfft(p, fApprox; timing)

        println(timing)
      end
    end
  end
  return nothing
end

nfft_performance_1()


function nfft_performance_2(N = 64, M = N*N*N)
  println("\n\n ##### nfft_performance_2 - multithreading ##### \n\n")

  m = 3; σ = 2.0
  timing = TimingStats()

  let x = Float32.(rand(3,M) .- 0.5), fHat = ComplexF32.(rand(M)*1im)

    for pre in [NFFT.LUT, NFFT.FULL] 
      for threading in [true, false]
        NFFT._use_threads[] = threading

        @info "* precomputation = $pre threading = $threading"
        p = plan_nfft(x, (N,N,N); m, σ, precompute=pre, timing, fftflags=FFTW.MEASURE)
        fApprox = nfft_adjoint(p, fHat; timing)
        nfft(p, fApprox; timing)

        println(timing)
      end
    end
  end
end


nfft_performance_2()
#nfft_performance_2(128,46_000)



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
  tadjoint = @elapsed fApprox = nfft_adjoint!(p, fHat, f; timing)
  ttrafo = @elapsed nfft!(p, fApprox, fHat; timing)

  @info tpre, ttrafo, tadjoint

  println(timing)

end


