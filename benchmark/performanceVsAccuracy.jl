using NFFT, DataFrames, LinearAlgebra, LaTeXStrings, DelimitedFiles
using BenchmarkTools
using Plots; pgfplotsx()

include("../Wrappers/NFFT3.jl")
include("../Wrappers/FINUFFT.jl")

const packagesCtor = [NFFTPlan,  NFFT3Plan, FINUFFTPlan]
const packagesStr = [ "NFFT.jl/TENSOR",  "NFFT3/TENSOR", "FINUFFT"]
const precomp = [NFFT.TENSOR, NFFT.TENSOR, NFFT.LUT]
const blocking = [true, true, true]

const benchmarkTime = [1, 1]

NFFT.FFTW.set_num_threads(Threads.nthreads())
ccall(("omp_set_num_threads",NFFT3.lib_path_nfft),Nothing,(Int64,),convert(Int64,Threads.nthreads()))
@info ccall(("nfft_get_num_threads",NFFT3.lib_path_nfft),Int64,())
NFFT._use_threads[] = (Threads.nthreads() > 1)


const σs = [2.0] 
const ms = 3:8
const NBase = [4*4096, 128, 32]
const Ds = 1:3

function nfft_accuracy_comparison(Ds=1:3)
  println("\n\n ##### nfft_performance vs accuracy ##### \n\n")

  df = DataFrame(Package=String[], D=Int[], M=Int[], N=Int[], m = Int[], σ=Float64[],
                   ErrorTrafo=Float64[], ErrorAdjoint=Float64[], 
                   TimeTrafo=Float64[], TimeAdjoint=Float64[] )  

  for D in Ds
    N = ntuple(d->NBase[D], D)
    M = prod(N)
    
    for σ in σs
      for m in ms
        @info "m=$m D=$D σ=$σ "
        x = rand(D,M) .- 0.5
        fHat = randn(ComplexF64, M)
        fApprox = randn(ComplexF64, N)
        gHatApprox = randn(ComplexF64, M)

        # ground truth (numerical)
        pNDFT = NDFTPlan(x, N)
        f = adjoint(pNDFT) * fHat
        gHat = pNDFT * f

        for pl = 1:length(packagesStr)

          planner = packagesCtor[pl]
          p = planner(x, N; m, σ, precompute=precomp[pl], blocking=blocking[pl])

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

          if planner == FINUFFTPlan 
            # This extracts the raw trafo timing that the FINUFFTPlan caches internally
            ttrafo = p.timeTrafo
            tadjoint = p.timeAdjoint
          end

          push!(df, (packagesStr[pl], D, M, N[D], m, σ, etrafo, eadjoint, ttrafo, tadjoint))

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
  
  titleTrafo = L"\textrm{NFFT}, \textrm{%$(D)D}"
  titleAdjoint = L"\textrm{NFFT}^H, \textrm{%$(D)D}"

  colors = [:black, :orange, :green, :brown, :gray, :blue, :purple, :yellow ]
  ls = [:solid, :dashdot, :dash, :solid, :dash, :solid, :dash, :solid]
  shape = [:xcross, :circle, :xcross, :circle, :xcross, :xcross, :circle]

  p1 = plot(df1_[df1_.Package.==packagesStr[1],:ErrorTrafo], 
            df1_[df1_.Package.==packagesStr[1],:TimeTrafo], # yscale = :log10, #ylims=(1e-4,1e-2),
            xscale = :log10, label=packagesStr[1], lw=2, xlabel = "Relative Error", ylabel="Runtime / s",
            legend = (:topright), title=titleTrafo, shape=:circle, c=:black)

  for p=2:length(packagesStr)      
    plot!(p1, df1_[df1_.Package.==packagesStr[p],:ErrorTrafo], 
          df1_[df1_.Package.==packagesStr[p],:TimeTrafo], #yscale = :log10,
            xscale = :log10, label=packagesStr[p], lw=2, shape=shape[p], ls=ls[p], 
            c=colors[p], msc=colors[p], mc=colors[p], ms=5, msw=2)
  end

  p2 = plot(df1_[df1_.Package.==packagesStr[1],:ErrorAdjoint], 
            df1_[df1_.Package.==packagesStr[1],:TimeAdjoint], #yscale = :log10, #ylims=(1e-4,1e-2),
            xscale = :log10, label=packagesStr[1], lw=2, xlabel = "Relative Error", ylabel="Runtime / s",
            legend = (:topright), title=titleAdjoint, shape=:circle, c=:black)

  for p=2:length(packagesStr)      
    plot!(p2, df1_[df1_.Package.==packagesStr[p],:ErrorAdjoint], 
          df1_[df1_.Package.==packagesStr[p],:TimeAdjoint], # yscale = :log10,
            xscale = :log10, label=packagesStr[p], lw=2, shape=shape[p], ls=ls[p], 
            c=colors[p], msc=colors[p], mc=colors[p], ms=5, msw=2)
  end


  p = plot(p1, p2, layout=(1,2), size=(800,300), dpi=200)

  savefig(p, "../docs/src/assets/performanceVsAccuracy_D$(D).svg")
  return p
end



#df = nfft_accuracy_comparison(Ds)
#writedlm("performanceVsAccuracy.csv", Iterators.flatten(([names(df)], eachrow(df))), ',')

data, header = readdlm("performanceVsAccuracy.csv", ',', header=true);
df = DataFrame(data, vec(header))

for d=1:3
  plot_accuracy(df, d)
end








