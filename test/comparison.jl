using NFFT, DataFrames, Plots, LinearAlgebra
import NFFT3



function nfft_accuracy_comparison()
  println("\n\n ##### nfft_performance_comparison ##### \n\n")

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

df = nfft_accuracy_comparison()

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
plot_accuracy(df, 1)
plot_accuracy(df, 2)


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