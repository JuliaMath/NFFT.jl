using NFFT, DataFrames, LinearAlgebra, LaTeXStrings, DelimitedFiles
using Plots; pgfplotsx()

include("../Wrappers/NFFT3.jl")
include("../Wrappers/FINUFFT.jl")


#const packagesCtor = [NFFTPlan, NFFTPlan, NFFTPlan,  NFFT3Plan, NFFT3Plan, FINUFFTPlan]
#const packagesStr = ["NFFT.jl/FULL", "NFFT.jl/LUT", "NFFT.jl/TENSOR", "NFFT3/LUT", "NFFT3/TENSOR", "FINUFFT"]
#const packagesNoFinufft = ["NFFT.jl/FULL", "NFFT.jl/LUT", "NFFT.jl/TENSOR", "NFFT3/LUT", "NFFT3/TENSOR"]
#const precomp = [NFFT.FULL, NFFT.LUT, NFFT.TENSOR, NFFT.LUT, NFFT.TENSOR, NFFT.LUT]
#const blocking = [false, true, true, false, false, false, false]

const packagesCtor = [NFFTPlan,  NFFT3Plan, FINUFFTPlan]
const packagesStr = ["NFFT.jl",  "NFFT3", "FINUFFT"]
const packagesNoFinufft = ["NFFT.jl",  "NFFT3"]
const precomp = [NFFT.TENSOR,  NFFT.TENSOR, NFFT.LUT]
const blocking = [true, true, true]


const σs = range(1.25, 4, length=12)
const ms = 3:10
const NBase = [4096, 64, 16]
const Ds = 1:3

function nfft_accuracy_comparison(Ds, σs, ms)
  println("\n\n ##### nfft_accuracy_comparison ##### \n\n")

  df = DataFrame(Package=String[], D=Int[], M=Int[], N=Int[], m = Int[], σ=Float64[],
                   ErrorTrafo=Float64[], ErrorAdjoint=Float64[] )  

  for D in Ds
      N = ntuple(d->NBase[D], D)
      M = prod(N)
      
      for σ in σs
        for m in ms
          @info "m=$m D=$D σ=$σ "
          x = rand(D,M) .- 0.5
          fHat = randn(ComplexF64, M)

          # ground truth (numerical)
          pNDFT = NDFTPlan(x, N)
          f = adjoint(pNDFT) * fHat
          gHat = pNDFT * f

          for pl = 1:length(packagesStr)
            planner = packagesCtor[pl]
            if planner != FINUFFT || σ == 2.0 # FINUFFT is not included in sigma sweep
              p = planner(x, N; m, σ, precompute=precomp[pl], blocking=blocking[pl])

              fApprox = adjoint(p) * fHat
              eadjoint = norm(f[:] - fApprox[:]) / norm(f[:])

              gHatApprox = p * f
              etrafo = norm(gHat[:] - gHatApprox[:]) / norm(gHat[:])
              
              push!(df, (packagesStr[pl], D, M, N[D], m, σ, etrafo, eadjoint))
            end
        end
      end
    end
  end
  return df
end

function plot_accuracy_m(df, D=1)

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
            legend = (:topright), title=L"\textrm{NFFT}", shape=:circle, c=:black)

  for p=2:length(packagesStr)      
    plot!(p1, ms, df1_[df1_.Package.==packagesStr[p],:ErrorTrafo], 
            yscale = :log10, label=packagesStr[p], lw=2, shape=shape[p], ls=ls[p], 
            c=colors[p], msc=colors[p], mc=colors[p], ms=5, msw=2)
  end

  p2 = plot(ms, df1_[df1_.Package.==packagesStr[1],:ErrorAdjoint], 
            yscale = :log10, label=packagesStr[1], lw=2, xlabel = "m", ylabel="Relative Error",
            legend = (:topright), title=L"\textrm{NFFT}^H", shape=:circle, c=:black)

  for p=2:length(packagesStr)      
    plot!(p2, ms, df1_[df1_.Package.==packagesStr[p],:ErrorAdjoint], 
            yscale = :log10, label=packagesStr[p], lw=2, shape=shape[p], ls=ls[p], 
            c=colors[p], msc=colors[p], mc=colors[p], ms=5, msw=2)
  end

  p = plot(p1, p2, layout=(1,2), size=(800,300), dpi=200)
  #p = plot(p1, layout=(1,2), size=(800,450), dpi=200)

  savefig(p, "../docs/src/assets/accuracy_m_D$(D).svg")
  return p
end


function plot_accuracy_sigma(df, D=1)

  σs = range(1.25, 4, length=12)

  df1_ = df[df.m.==4 .&& df.D.==D,:]

  Plots.scalefontsizes()
  Plots.scalefontsizes(1.5)
  

  colors = [:black, :orange, :green, :brown, :gray, :blue, :purple, :yellow ]
  ls = [:solid, :dashdot, :dash, :solid, :dash, :solid, :dash, :solid]
  shape = [:xcross, :circle, :xcross, :circle, :xcross, :xcross, :circle]

  p1 = plot(σs, df1_[df1_.Package.==packagesNoFinufft[1],:ErrorTrafo], 
            yscale = :log10, label=packagesNoFinufft[1], lw=2, xlabel = L"\sigma", ylabel="Relative Error",
            legend = (:topright), title=L"\textrm{NFFT}", shape=:circle, c=:black)

  for p=2:length(packagesNoFinufft)      
    plot!(p1, σs, df1_[df1_.Package.==packagesNoFinufft[p],:ErrorTrafo], 
            yscale = :log10, label=packagesNoFinufft[p], lw=2, shape=shape[p], ls=ls[p], 
            c=colors[p], msc=colors[p], mc=colors[p], ms=5, msw=2)
  end

  p2 = plot(σs, df1_[df1_.Package.==packagesNoFinufft[1],:ErrorAdjoint], 
            yscale = :log10, label=packagesNoFinufft[1], lw=2, xlabel = L"\sigma", ylabel="Relative Error",
            legend = (:topright), title=L"\textrm{NFFT}^H", shape=:circle, c=:black)

  for p=2:length(packagesNoFinufft)      
    plot!(p2, σs, df1_[df1_.Package.==packagesNoFinufft[p],:ErrorAdjoint], 
            yscale = :log10, label=packagesNoFinufft[p], lw=2, shape=shape[p], ls=ls[p], 
            c=colors[p], msc=colors[p], mc=colors[p], ms=5, msw=2)
  end


  #=p2 = plot(σs, df2_[df2_.Package.=="NFFT.jl",:ErrorTrafo], 
        yscale = :log10, label="NFFT.jl", lw=2, xlabel = L"\sigma",  ylabel="Relative Error",
        legend = :none, title=L"m = 8",shape=:circle, c=:black)
  plot!(p2, σs, df2_[df2_.Package.=="NFFT3",:ErrorTrafo], 
    yscale = :log10, label="NFFT3", lw=2, shape=:xcross, ls=:solid, 
    c=:gray, msc=:gray, mc=:gray, ms=6, msw=3)=#


  p = plot(p1, p2, layout=(1,2), size=(800,300), dpi=200)

  savefig(p, "../docs/src/assets/accuracy_sigma_D$(D).svg")
  return p
end


dfm = nfft_accuracy_comparison(2, [2.0], ms)
dfσ = nfft_accuracy_comparison(2, σs, [4])

writedlm("accuracy_m.csv", Iterators.flatten(([names(dfm)], eachrow(dfm))), ',')
writedlm("accuracy_sigma.csv", Iterators.flatten(([names(dfσ)], eachrow(dfσ))), ',')

data, header = readdlm("accuracy_m.csv", ',', header=true);
dfm = DataFrame(data, vec(header))
data, header = readdlm("accuracy_sigma.csv", ',', header=true);
dfσ = DataFrame(data, vec(header))


plot_accuracy_m(dfm, 2)
plot_accuracy_sigma(dfσ, 2)








