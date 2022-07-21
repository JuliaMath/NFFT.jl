using NFFT, DataFrames, LinearAlgebra, LaTeXStrings, DelimitedFiles
using BenchmarkTools
using Plots; pgfplotsx()


const packagesCtor = [NFFTPlan, NFFTPlan]
const packagesStr = ["NFFT.jl/POLY/NonBlock", "NFFT.jl/POLY/Block"]
const packagesStrShort = ["Regular", "Block Partitioning"]
const precomp = [NFFT.POLYNOMIAL, NFFT.POLYNOMIAL]
const blocking = [false, true]

const benchmarkTime = [10, 10]

NFFT._use_threads[] = (Threads.nthreads() > 1)

const σs = [2.0] 
const ms = 3:8
const NBase = [65536, 256, 32] # [4*4096, 128, 32]
const Ds = 2

function nfft_accuracy_comparison(Ds=1:3)
  println("\n\n ##### nfft_performance blocking ##### \n\n")

  df = DataFrame(Package=String[], D=Int[], J=Int[], N=Int[], m = Int[], σ=Float64[],
                   ErrorTrafo=Float64[], ErrorAdjoint=Float64[], 
                   TimeTrafo=Float64[], TimeAdjoint=Float64[] )  

  for D in Ds
    @info "### Dimension D=$D ###"
    N = ntuple(d->NBase[D], D)
    J = prod(N)
    
    k = rand(D,J) .- 0.5
    fHat = randn(ComplexF64, J)
    fApprox = randn(ComplexF64, N)
    gHatApprox = randn(ComplexF64, J)

    # ground truth (numerical)
    pNDFT = NDFTPlan(k, N)
    f = adjoint(pNDFT) * fHat
    gHat = pNDFT * f

    for σ in σs
      for m in ms
        @info "m=$m D=$D σ=$σ "

        for pl = 1:length(packagesStr)
          planner = packagesCtor[pl]
          p = planner(k, N; m, σ, precompute=precomp[pl], blocking=blocking[pl])

          @info "Adjoint accuracy: $(packagesStr[pl])"
          mul!(fApprox, adjoint(p), fHat)
          eadjoint = norm(f[:] - fApprox[:]) / norm(f[:])

          @info "Adjoint benchmark: $(packagesStr[pl])"
          BenchmarkTools.DEFAULT_PARAMETERS.seconds = benchmarkTime[1] 
          b = @benchmark mul!($fApprox, $(adjoint(p)), $fHat)
          tadjoint = minimum(b).time / 1e9

          @info "Trafo accuracy: $(packagesStr[pl])"
          mul!(gHatApprox, p, f)
          etrafo = norm(gHat[:] - gHatApprox[:]) / norm(gHat[:])

          @info "Trafo benchmark: $(packagesStr[pl])"
          BenchmarkTools.DEFAULT_PARAMETERS.seconds = benchmarkTime[2]
          b = @benchmark mul!($gHatApprox, $p, $f)
          ttrafo = minimum(b).time / 1e9

          push!(df, (packagesStr[pl], D, J, N[D], m, σ, etrafo, eadjoint, ttrafo, tadjoint))

        end
      end
    end
  end
  return df
end



function plot_accuracy(df, D=1)

  df1_ = df[df.σ.==2.0 .&& df.D.==D,:]
  Plots.scalefontsizes()
  Plots.scalefontsizes(1.5)
  
  titleTrafo = L"\textrm{NFFT}" #, \textrm{%$(D)D}
  titleAdjoint = L"\textrm{NFFT}^H" #, \textrm{%$(D)D}

  colors = [:black, :orange, :blue, :green, :brown, :gray, :blue, :purple, :yellow ]
  ls = [:solid, :dashdot, :solid, :solid, :solid, :dash, :solid, :dash, :solid]
  shape = [:circle, :circle, :circle, :xcross, :circle, :xcross, :xcross, :circle]

  maxTimeTrafo = maximum(df1_[:,:TimeTrafo])
  maxTimeAdjoint = maximum(df1_[:,:TimeAdjoint])

  p1 = plot(df1_[df1_.Package.==packagesStr[1],:ErrorTrafo], 
            df1_[df1_.Package.==packagesStr[1],:TimeTrafo], ylims=(0.0,maxTimeTrafo),
            xscale = :log10, label=packagesStrShort[1], lw=2, xlabel = "Relative Error", ylabel="Runtime / s",
            legend = (:topright), title=titleTrafo, shape=:circle, c=:black)

  for p=2:length(packagesStr)      
    plot!(p1, df1_[df1_.Package.==packagesStr[p],:ErrorTrafo], 
          df1_[df1_.Package.==packagesStr[p],:TimeTrafo], 
            xscale = :log10, label=packagesStrShort[p], lw=2, shape=shape[p], ls=ls[p], 
            c=colors[p], msc=colors[p], mc=colors[p], ms=4, msw=2)
  end

  p2 = plot(df1_[df1_.Package.==packagesStr[1],:ErrorAdjoint], 
            df1_[df1_.Package.==packagesStr[1],:TimeAdjoint], ylims=(0.0,maxTimeAdjoint),
            xscale = :log10,  lw=2, xlabel = "Relative Error", #ylabel="Runtime / s", #label=packagesStr[1],
            legend = nothing, title=titleAdjoint, shape=:circle, c=:black)

  for p=2:length(packagesStr)      
    plot!(p2, df1_[df1_.Package.==packagesStr[p],:ErrorAdjoint], 
          df1_[df1_.Package.==packagesStr[p],:TimeAdjoint], 
            xscale = :log10,  lw=2, shape=shape[p], ls=ls[p], #label=packagesStr[p],
            c=colors[p], msc=colors[p], mc=colors[p], ms=4, msw=2)
  end


  p = plot(p1, p2, layout=(1,2), size=(800,300), dpi=200)

  mkpath("./img/")
  savefig(p, "./img/performanceBlocking_D$(D).pdf")
  return p
end



df = nfft_accuracy_comparison(Ds)
writedlm("data/performanceBlocking.csv", Iterators.flatten(([names(df)], eachrow(df))), ',')

data, header = readdlm("data/performanceBlocking.csv", ',', header=true);
df = DataFrame(data, vec(header))

for d in Ds
  plot_accuracy(df, d)
end








