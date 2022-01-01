using NFFT, DataFrames, Plots, LinearAlgebra, FFTW, BenchmarkTools
import NFFT3



function nfft_accuracy_comparison()
  println("\n\n ##### nfft_accuracy_comparison ##### \n\n")

  df = DataFrame(Package=String[], D=Int[], M=Int[], N=Int[], m = Int[], sigma=Float64[],
                   ErrorTrafo=Float64[], ErrorAdjoint=Float64[] )  
  N = [256, 64]

  for D = 1:2
  
      NN = ntuple(d->N[D], D)
      M = prod(NN)
      
      for sigma in [1.25, 1.5, 2.0]
        for m = 1:14
          @info "D=$D  sigma=$sigma  m=$m "
          x = rand(D,M) .- 0.5
          fHat = randn(ComplexF64, M)

          p = plan_nfft(x, NN, m, sigma; precompute=NFFT.FULL)
          f = ndft_adjoint(p, fHat)
          fApprox = nfft_adjoint(p, fHat)
          eadjoint = norm(f[:] - fApprox[:]) / norm(f[:])

          gHat = ndft(p, f)
          gHatApprox = nfft(p, f)
          etrafo = norm(gHat[:] - gHatApprox[:]) / norm(gHat[:])
          
          push!(df, ("NFFT.jl", D, M, N[D], m, sigma, etrafo, eadjoint))

          pnfft3 = NFFT3.NFFT(NN, M, Int32.(p.n), m) 
          pnfft3.x = (D==1) ? vec(x) : x
          pnfft3.f = fHat
          NFFT3.nfft_adjoint(pnfft3)
          fApprox = reshape(pnfft3.fhat,reverse(NN)...)
          # switch from column major to row major format
          fApprox = (D==1) ? vec(fApprox) : vec(collect(permutedims(fApprox,D:-1:1)))   
          eadjoint = norm(f[:] - fApprox[:]) / norm(f[:])

          # switch from column major to row major format
          pnfft3.fhat = (D==1) ? vec(f) : vec(collect(permutedims(f,D:-1:1))) 
          NFFT3.nfft_trafo(pnfft3)
          gHatApprox = pnfft3.f
          etrafo = norm(gHat[:] - gHatApprox[:]) / norm(gHat[:])
          
          push!(df, ("NFFT3", D, M, N[D], m, sigma, etrafo, eadjoint))
      end
    end
  end
  return df
end



function plot_accuracy(df, D=1)

  sigmas = [1.25, 1.5, 2.0]

  plots = Matrix{Any}(undef, length(sigmas), 2)

  m = 1:14

  for (i,sigma) in enumerate(sigmas)
    df_ = df[df.sigma.==sigma .&& df.D.==D,:]

    p1 = plot(m, df_[df_.Package.=="NFFT.jl",:ErrorTrafo], 
              yscale = :log10, label="NFFT.jl", lw=2, xlabel = "m", title="Trafo σ=$(sigma)")
    plot!(p1, m, df_[df_.Package.=="NFFT3",:ErrorTrafo], 
          yscale = :log10, label="NFFT3", lw=2)

    plots[i,1] = p1

    p2 = plot(m, df_[df_.Package.=="NFFT.jl",:ErrorAdjoint], 
          yscale = :log10, label="NFFT.jl", lw=2, xlabel = "m", title="Adjoint σ=$(sigma)")
    plot!(p2, m, df_[df_.Package.=="NFFT3",:ErrorAdjoint], 
      yscale = :log10, label="NFFT3", lw=2)

    plots[i,2] = p2
  end

  p = plot(plots..., layout=(2,length(sigmas)))
  savefig(p, "accuracy_D$(D).png")
end

#df = nfft_accuracy_comparison()
#plot_accuracy(df, 1)
#plot_accuracy(df, 2)

const K = 100000

FFTW.set_num_threads(Threads.nthreads())
ccall(("omp_set_num_threads",NFFT3.lib_path_nfft),Nothing,(Int64,),convert(Int64,Threads.nthreads()))
@info ccall(("nfft_get_num_threads",NFFT3.lib_path_nfft),Int64,())
if Threads.nthreads() == 1
  NFFT._use_threads[] = false
end

function nfft_performance_comparison(m = 5, sigma = 2.0)
  println("\n\n ##### nfft_performance_comparison ##### \n\n")

  df = DataFrame(Package=String[], D=Int[], M=Int[], N=Int[], 
                   Undersampled=Bool[], Pre=String[], m = Int[], sigma=Float64[],
                   TimePre=Float64[], TimeTrafo=Float64[], TimeAdjoint=Float64[] )  

  preString = ["LUT", "FULL", "FULL_LUT"]
  preNFFTjl = [NFFT.LUT, NFFT.FULL, NFFT.FULL_LUT]
  N = [collect(4096* (4 .^(0:3))),collect(128* (2 .^ (0:3))),[32,48,64,72]]

  for D = 2:2
    for U = 4:4
      NN = ntuple(d->N[D][U], D)
      M = prod(NN) #*2 #÷ 10

      for pre = 1:2

        @info D, NN, M, pre
        
        T = Float64

        x = T.(rand(T,D,M) .- 0.5)
        fHat = randn(Complex{T}, M)

        tpre = @elapsed p = plan_nfft(x, NN, m, sigma, :kaiser_bessel, K; precompute=preNFFTjl[pre], sortNodes=false)
        f = similar(fHat, p.N)
        tadjoint = @elapsed fApprox = nfft_adjoint!(p, fHat, f)
        ttrafo = @elapsed nfft!(p, fApprox, fHat)
        
        push!(df, ("NFFT.jl", D, M, N[D][U], false, preString[pre], m, sigma,
                   tpre, ttrafo, tadjoint))

        prePsi = pre == 1 ? NFFT3.PRE_LIN_PSI : NFFT3.PRE_FULL_PSI

        f1 = UInt32(
          NFFT3.PRE_PHI_HUT |
          prePsi |
          NFFT3.MALLOC_X |
          NFFT3.MALLOC_F_HAT |
          NFFT3.MALLOC_F |
          NFFT3.FFTW_INIT |
          NFFT3.FFT_OUT_OF_PLACE |
          #NFFT3.NFCT_SORT_NODES |
          NFFT3.NFCT_OMP_BLOCKWISE_ADJOINT
           )

        f2 = UInt32(NFFT3.FFTW_ESTIMATE | NFFT3.FFTW_DESTROY_INPUT)

        tpre = @elapsed begin
          pnfft3 = NFFT3.NFFT(NN, M, Int32.(p.n), m, f1, f2) 
          NFFT3.nfft_init(pnfft3)
        end 

        pnfft3.x = Float64.(x)
        pnfft3.fhat = vec(ComplexF64.(f))
        ttrafo = @elapsed NFFT3.nfft_trafo(pnfft3)
        tadjoint = @elapsed fApprox = NFFT3.nfft_adjoint(pnfft3)
        
        push!(df, ("NFFT3", D, M, N[D][U], false, preString[pre], m, sigma,
                    tpre, ttrafo, tadjoint))
          
      end
    end
  end
  return df
end

df = nfft_performance_comparison(3, 2.0)

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