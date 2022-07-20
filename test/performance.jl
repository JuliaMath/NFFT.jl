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
    let N = 2^19, J =N, k = rand(J) .- 0.5, fHat = rand(J)*1im
      for pre in [NFFT.LINEAR, NFFT.FULL]
        @info "* precomputation = $pre"
        p = plan_nfft(k, N; m, σ, precompute=pre, timing)
        fApprox = *(adjoint(p), fHat; timing)
        *(p, fApprox; timing)

        println(timing)
      end
    end

    @info "NFFT Performance Test 2D"
    let N = 1024, J =N*N, x2 = rand(2,J) .- 0.5, fHat = rand(J)*1im
      for pre in [NFFT.LINEAR, NFFT.FULL]
        @info "* precomputation = $pre"
        p = plan_nfft(x2, (N,N); m, σ, precompute=pre, timing)
        fApprox = *(adjoint(p), fHat; timing)
        *(p, fApprox; timing)

        println(timing)
      end
    end

    @info "NFFT Performance Test 3D"
    let N = 32, J =N*N*N, x3 = rand(3,J) .- 0.5, fHat = rand(J)*1im
      for pre in [NFFT.LINEAR, NFFT.FULL]
        @info "* precomputation = $pre"
        p = plan_nfft(x3, (N,N,N); m, σ, precompute=pre, timing)
        fApprox = *(adjoint(p), fHat; timing)
        *(p, fApprox; timing)

        println(timing)
      end
    end
  end
  return nothing
end

nfft_performance_1()


function nfft_performance_2(N = 64, J =N*N*N)
  println("\n\n ##### nfft_performance_2 - multithreading ##### \n\n")

  m = 3; σ = 2.0
  timing = TimingStats()

  let k =Float32.(rand(3,J) .- 0.5), fHat = ComplexF32.(rand(J)*1im)

    for pre in [NFFT.LINEAR, NFFT.FULL] 
      for threading in [true, false]
        NFFT._use_threads[] = threading

        @info "* precomputation = $pre threading = $threading"
        p = plan_nfft(k, (N,N,N); m, σ, precompute=pre, timing, fftflags=FFTW.MEASURE)
        fApprox = *(adjoint(p), fHat; timing)
        *(p, fApprox; timing)

        println(timing)
      end
    end
  end
end


nfft_performance_2()
#nfft_performance_2(128,46_000)





