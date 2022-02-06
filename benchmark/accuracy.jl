using NFFT, DataFrames, LinearAlgebra, LaTeXStrings, DelimitedFiles
using Plots; pgfplotsx()

include("../Wrappers/NFFT3.jl")
include("../Wrappers/FINUFFT.jl")

#const packagesCtor = [NFFTPlan, NFFTPlan, NFFTPlan, FINUFFTPlan, NFFT3Plan]
#const packagesStr = ["NFFT.jl/LUT","NFFT.jl/NonBlock","NFFT.jl/FULL","FINUFFT", "NFFT3"]
#const precomp = [NFFT.LUT, NFFT.LUT, NFFT.FULL, NFFT.LUT, NFFT.LUT]
#const blocking = [true, false, false, false, false]
const packagesCtor = [NFFTPlan, NFFTPlan, NFFTPlan,  NFFT3Plan, NFFT3Plan, FINUFFTPlan]
const packagesStr = ["NFFT.jl/FULL", "NFFT.jl/LUT", "NFFT.jl/TENSOR", "NFFT3/LUT", "NFFT3/TENSOR", "FINUFFTPlan"]
const precomp = [NFFT.FULL, NFFT.LUT, NFFT.TENSOR, NFFT.LUT, NFFT.TENSOR, NFFT.LUT]
const blocking = [false, true, true, false, false, false, false]


const σs = [2.0] #range(1.25, 4, length=12)
const ms = 3:10 
const LUTSize=2^14

function nfft_accuracy_comparison()
  println("\n\n ##### nfft_accuracy_comparison ##### \n\n")

  df = DataFrame(Package=String[], D=Int[], M=Int[], N=Int[], m = Int[], σ=Float64[],
                   ErrorTrafo=Float64[], ErrorAdjoint=Float64[] )  
  N = [256, 64]

  for D = 2:2
  
      NN = ntuple(d->N[D], D)
      M = prod(NN)
      
      for σ in σs
        for m in ms
          @info "D=$D  σ=$σ  m=$m "
          x = rand(D,M) .- 0.5
          fHat = randn(ComplexF64, M)

          for pl = 1:length(packagesStr)

            planner = packagesCtor[pl]
            p = planner(x, NN; m, σ, precompute=precomp[pl], LUTSize=LUTSize, blocking=blocking[pl])
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

  df1_ = df[df.σ.==2.0 .&& df.D.==D,:]
  #df2_ = df[df.m.==8 .&& df.D.==D,:]

  Plots.scalefontsizes()
  Plots.scalefontsizes(1.5)
  

  colors = [:black, :orange, :green, :brown, :gray, :blue, :purple, :yellow ]
  ls = [:solid, :dashdot, :dash, :solid, :dash, :solid, :dash, :solid]
  shape = [:xcross, :circle, :xcross, :circle, :xcross, :xcross, :circle]

  p1 = plot(ms, df1_[df1_.Package.==packagesStr[1],:ErrorTrafo], 
            yscale = :log10, label=packagesStr[1], lw=2, xlabel = "m", ylabel="Relative Error",
            legend = (:topright), title=L"\sigma = 2", shape=:circle, c=:black)

  for p=2:length(packagesStr)      
    plot!(p1, ms, df1_[df1_.Package.==packagesStr[p],:ErrorTrafo], 
            yscale = :log10, label=packagesStr[p], lw=2, shape=shape[p], ls=ls[p], 
            c=colors[p], msc=colors[p], mc=colors[p], ms=5, msw=2)
  end


  #=p2 = plot(σs, df2_[df2_.Package.=="NFFT.jl",:ErrorTrafo], 
        yscale = :log10, label="NFFT.jl", lw=2, xlabel = L"\sigma",  ylabel="Relative Error",
        legend = :none, title=L"m = 8",shape=:circle, c=:black)
  plot!(p2, σs, df2_[df2_.Package.=="NFFT3",:ErrorTrafo], 
    yscale = :log10, label="NFFT3", lw=2, shape=:xcross, ls=:solid, 
    c=:gray, msc=:gray, mc=:gray, ms=6, msw=3)=#


  #p = plot(p1, p2, layout=(1,2), size=(800,300), dpi=200)
  p = plot(p1, layout=(1,2), size=(800,450), dpi=200)

  savefig(p, "../docs/src/assets/accuracy_D$(D).svg")
  return p
end




#df = nfft_accuracy_comparison()
#writedlm("accuracy.csv", Iterators.flatten(([names(df)], eachrow(df))), ',')

data, header = readdlm("accuracy.csv", ',', header=true);
df = DataFrame(data, vec(header))


plot_accuracy(df, 2)








