using NFFT
using FFTW
import NFFT3
using DataFrames

FFTW.set_num_threads(Threads.nthreads())

### performance test ###
function nfft_performance_1()
  println("\n\n ##### nfft_performance_1 - simple ##### \n\n")

  timing = TimingStats()

  let m = 3, sigma = 2.0
    @info "NFFT Performance Test 1D"
    let N = 2^19, M = N, x = rand(M) .- 0.5, fHat = rand(M)*1im
      for pre in [NFFT.LUT, NFFT.FULL]
        @info "* precomputation = $pre"
        p = plan_nfft(x, N, m, sigma; precompute=pre, timing)
        fApprox = nfft_adjoint(p, fHat; timing)
        nfft(p, fApprox; timing)

        println(timing)
      end
    end

    @info "NFFT Performance Test 2D"
    let N = 1024, M = N*N, x2 = rand(2,M) .- 0.5, fHat = rand(M)*1im
      for pre in [NFFT.LUT, NFFT.FULL]
        @info "* precomputation = $pre"
        p = plan_nfft(x2, (N,N), m, sigma; precompute=pre, timing)
        fApprox = nfft_adjoint(p, fHat; timing)
        nfft(p, fApprox; timing)

        println(timing)
      end
    end

    @info "NFFT Performance Test 3D"
    let N = 32, M = N*N*N, x3 = rand(3,M) .- 0.5, fHat = rand(M)*1im
      for pre in [NFFT.LUT, NFFT.FULL]
        @info "* precomputation = $pre"
        p = plan_nfft(x3, (N,N,N), m, sigma; precompute=pre, timing)
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

  m = 3; sigma = 2.0
  timing = TimingStats()

  let x3 = Float32.(rand(3,M) .- 0.5), fHat = ComplexF32.(rand(M)*1im)

    for pre in [NFFT.LUT, NFFT.FULL, NFFT.FULL_LUT] 
      for threading in [true, false]
        NFFT._use_threads[] = threading

        @info "* precomputation = $pre threading = $threading"
        p = plan_nfft(x3, (N,N,N), m, sigma; precompute=pre, timing, flags=FFTW.MEASURE)
        fApprox = nfft_adjoint(p, fHat; timing)
        nfft(p, fApprox; timing)

        println(timing)
      end
    end
  end
end


nfft_performance_2()
#nfft_performance_2(128,46_000)




function nfft_performance_comparison(m = 5, sigma = 2.0)
  println("\n\n ##### nfft_performance_comparison ##### \n\n")

  df = DataFrame(Package=String[], D=Int[], M=Int[], N=Int[], 
                   Undersampled=Bool[], Pre=String[], m = Int[], sigma=Float64[],
                   TimePre=Float64[], TimeTrafo=Float64[], TimeAdjoint=Float64[] )  

  preString = ["LUT", "FULL"]
  preNFFTjl = [NFFT.LUT, NFFT.FULL]
  N = [collect(4096* (4 .^(0:3))),collect(64* (2 .^ (0:3))),[32,48,64,72]]

  for D = 2:3
    for U = 1:4
      NN = ntuple(d->N[D][U], D)
      M = prod(NN)

      for pre = 1:2

        @info D, NN, M, pre
        
        x = rand(D,M) .- 0.5
        fHat = randn(ComplexF64, M)

        tpre = @elapsed p = plan_nfft(x, NN, m, sigma; precompute=preNFFTjl[pre])
        tadjoint = @elapsed fApprox = nfft_adjoint(p, fHat)
        ttrafo = @elapsed nfft(p, fApprox)
        
        push!(df, ("NFFT.jl", D, M, N[D][U], false, preString[pre], m, sigma,
                   tpre, ttrafo, tadjoint))


        
        tpre = @elapsed pnfft3 = NFFT3.NFFT(NN, M, Int32.(p.n), m) 
        pnfft3.x = x
        pnfft3.fhat = fHat
        ttrafo = @elapsed NFFT3.nfft_trafo(pnfft3)
        tadjoint = @elapsed fApprox = NFFT3.nfft_adjoint(pnfft3)
        
        push!(df, ("NFFT3", D, M, N[D][U], false, preString[pre], m, sigma,
                    tpre, ttrafo, tadjoint))
          
      end
    end
  end
  return df
end

# writedlm("test.csv", Iterators.flatten(([names(iris)], eachrow(iris))), ',')
#
#  julia> using DelimitedFiles, DataFrames
#
# julia> data, header = readdlm(joinpath(dirname(pathof(DataFrames)),
# "..", "docs", "src", "assets", "iris.csv"),
# ',', header=true);
#
#julia> iris_raw = DataFrame(data, vec(header))
#
#