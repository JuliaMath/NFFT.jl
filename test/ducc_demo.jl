using NFFT
using FFTW
import NFFT3
using DataFrames
using LinearAlgebra
include("../Wrappers/ducc0_nufft.jl")

### performance test ###
function nfft_performance_1()
  println("\n\n ##### nfft_performance_1 - simple ##### \n\n")

  timing = TimingStats()

  let m = 8, σ = 2.0
    @info "NFFT Performance Test 1D"
    let N = 2^19, J =N, k = rand(J) .- 0.5, fHat = rand(J)*1im
      for pre in [NFFT.POLYNOMIAL]
        @info "* precomputation = $pre"
        p = plan_nfft(k, N; m, σ, precompute=pre, timing)
        fApprox = *(adjoint(p), fHat; timing)
        foo = *(p, fApprox; timing)

        println(timing)
        x1 = Array{Float64,2}(undef, 1,J)
        x1[1,:] = k
        # use an epsilon that roughly matches m=8, sigma=2
        res=ducc_nu2u(x1, fHat, (N,), epsilon=1e-9, nthreads=1, verbosity=1)
        println("difference L2 norm of transform: ", norm(res[:] - fApprox[:]) / norm(fApprox[:]))
        res2=ducc_u2nu(x1, res, epsilon=1e-9, nthreads=1, verbosity=1)
        println("difference L2 norm of adjoint transform: ", norm(res2[:] - foo[:]) / norm(foo[:]))
      end
    end

    @info "NFFT Performance Test 2D"
    let N = 1024, J =N*N, x2 = rand(2,J) .- 0.5, fHat = rand(J)*1im
      for pre in [NFFT.POLYNOMIAL]
        @info "* precomputation = $pre"
        p = plan_nfft(x2, (N,N); m, σ, precompute=pre, timing)
        fApprox = *(adjoint(p), fHat; timing)
        foo = *(p, fApprox; timing)

        println(timing)
        # use an epsilon that roughly matches m=8, sigma=2
        res=ducc_nu2u(x2, fHat, (N,N), epsilon=1e-9, nthreads=1, verbosity=1)
        println("difference L2 norm of transform: ", norm(res[:] - fApprox[:]) / norm(fApprox[:]))
        res2=ducc_u2nu(x2, res, epsilon=1e-9, nthreads=1, verbosity=1)
        println("difference L2 norm of adjoint transform: ", norm(res2[:] - foo[:]) / norm(foo[:]))
      end
    end

    @info "NFFT Performance Test 3D"
    let N = 32, J =N*N*N, x3 = rand(3,J) .- 0.5, fHat = rand(J)*1im
      for pre in [NFFT.POLYNOMIAL]
        @info "* precomputation = $pre"
        p = plan_nfft(x3, (N,N,N); m, σ, precompute=pre, timing)
        fApprox = *(adjoint(p), fHat; timing)
        foo = *(p, fApprox; timing)

        println(timing)
        res=ducc_nu2u(x3, fHat, (N,N,N), epsilon=1e-9, nthreads=1, verbosity=1)
        println("difference L2 norm of transform: ", norm(res[:] - fApprox[:]) / norm(fApprox[:]))
        res2=ducc_u2nu(x3, res, epsilon=1e-9, nthreads=1, verbosity=1)
        println("difference L2 norm of adjoint transform: ", norm(res2[:] - foo[:]) / norm(foo[:]))
      end
    end
  end
  return nothing
end

nfft_performance_1()
nfft_performance_1()
nfft_performance_1()
nfft_performance_1()




