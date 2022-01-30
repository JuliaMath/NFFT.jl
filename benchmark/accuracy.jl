using NFFT, DataFrames, LinearAlgebra, LaTeXStrings, DelimitedFiles
using Plots; pgfplotsx()

include("../Wrappers/NFFT3.jl")
include("../Wrappers/FINUFFT.jl")

const packagesCtor = [NFFTPlan, FINUFFTPlan, NFFT3Plan]
const packagesStr = ["NFFT.jl","FINUFFT", "NFFT3"]

function nfft_accuracy_comparison()
  println("\n\n ##### nfft_accuracy_comparison ##### \n\n")

  df = DataFrame(Package=String[], D=Int[], M=Int[], N=Int[], m = Int[], σ=Float64[],
                   ErrorTrafo=Float64[], ErrorAdjoint=Float64[] )  
  N = [256, 64]

  for D = 2:2
  
      NN = ntuple(d->N[D], D)
      M = prod(NN)
      
      for σ in range(1.25, 4, length=12)
        for m = 2:14
          @info "D=$D  σ=$σ  m=$m "
          x = rand(D,M) .- 0.5
          fHat = randn(ComplexF64, M)

          for pl = 1:length(packagesStr)

            planner = packagesCtor[pl]

            p = planner(x, NN; m, σ, precompute=NFFT.LUT)
            pNDFT = NDFTPlan(x, NN)
            f = adjoint(pNDFT) * fHat
            fApprox = adjoint(p) * fHat
            eadjoint = norm(f[:] - fApprox[:]) / norm(f[:])

            gHat = pNDFT * f
            gHatApprox = p * f
            etrafo = norm(gHat[:] - gHatApprox[:]) / norm(gHat[:])
            
            push!(df, (packagesStr[pl], D, M, N[D], m, σ, etrafo, eadjoint))

        end
      end
    end
  end
  return df
end



function plot_accuracy(df, D=1)

  σs = range(1.25, 4, length=12)

  m = 2:14

  df1_ = df[df.σ.==σs[4] .&& df.D.==D,:]
  df2_ = df[df.m.==8 .&& df.D.==D,:]

  Plots.scalefontsizes()
  Plots.scalefontsizes(1.5)
  

  p1 = plot(m, df1_[df1_.Package.=="NFFT.jl",:ErrorTrafo], 
            yscale = :log10, label="NFFT.jl", lw=2, xlabel = "m", ylabel="Relative Error",
            legend = (:topright), title=L"\sigma = 2", shape=:circle, c=:black)
  plot!(p1, m, df1_[df1_.Package.=="NFFT3",:ErrorTrafo], 
        yscale = :log10, label="NFFT3", lw=2, shape=:xcross, ls=:solid, 
        c=:gray, msc=:gray, mc=:gray, ms=6, msw=3)

  plot!(p1, m, df1_[df1_.Package.=="FINUFFT",:ErrorTrafo], 
        yscale = :log10, label="FINUFFT", lw=2, shape=:xcross, ls=:solid, 
        c=:blue, msc=:blue, mc=:blue, ms=6, msw=3)

  p2 = plot(σs, df2_[df2_.Package.=="NFFT.jl",:ErrorTrafo], 
        yscale = :log10, label="NFFT.jl", lw=2, xlabel = L"\sigma",  ylabel="Relative Error",
        legend = :none, title=L"m = 8",shape=:circle, c=:black)
  plot!(p2, σs, df2_[df2_.Package.=="NFFT3",:ErrorTrafo], 
    yscale = :log10, label="NFFT3", lw=2, shape=:xcross, ls=:solid, 
    c=:gray, msc=:gray, mc=:gray, ms=6, msw=3)


  p = plot(p1, p2, layout=(1,2), size=(800,300), dpi=200)

  savefig(p, "../docs/src/assets/accuracy_D$(D).svg")
  return p
end




df = nfft_accuracy_comparison()
writedlm("accuracy.csv", Iterators.flatten(([names(df)], eachrow(df))), ',')

data, header = readdlm("accuracy.csv", ',', header=true);
df = DataFrame(data, vec(header))


plot_accuracy(df, 2)


















function plot_accuracy_(df, D=2)

  σs = [1.25, 1.5, 2.0]

  plots = Matrix{Any}(undef, length(σs), 2)

  m = 1:14

  for (i,σ) in enumerate(σs)
    df_ = df[df.σ.==σ .&& df.D.==D,:]
 
    showlegend = (i==1)

    p1 = plot(m, df_[df_.Package.=="NFFT.jl",:ErrorTrafo], 
              yscale = :log10, label="NFFT.jl", lw=2, xlabel = "m", title="Trafo σ=$(σ)",
              legend = showlegend ? (:top) : (:none), shape=:circle)
    plot!(p1, m, df_[df_.Package.=="NFFT3",:ErrorTrafo], 
          yscale = :log10, label="NFFT3", lw=2, shape=:xcross)

    plots[i,1] = p1

    p2 = plot(m, df_[df_.Package.=="NFFT.jl",:ErrorAdjoint], 
          yscale = :log10, label="NFFT.jl", lw=2, xlabel = "m", title="Adjoint σ=$(σ)",
          legend = :none)
    plot!(p2, m, df_[df_.Package.=="NFFT3",:ErrorAdjoint], 
      yscale = :log10, label="NFFT3", lw=2)

    plots[i,2] = p2
  end

  p = plot(plots..., layout=(2,length(σs)))
  
  savefig(p, "accuracy_D$(D).png")
end
