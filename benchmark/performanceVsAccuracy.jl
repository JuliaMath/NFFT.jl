using NFFT, DataFrames, LinearAlgebra, LaTeXStrings, DelimitedFiles, Serialization
using BenchmarkTools
using Plots; pgfplotsx()
using Plots.Measures

include("../Wrappers/NFFT3.jl")
include("../Wrappers/FINUFFT.jl")
include("../Wrappers/DUCC0.jl")

const packagesCtor = [NFFTPlan, NFFTPlan, NFFT3Plan, FINUFFTPlan, Ducc0NufftPlan]
const packagesStr = [ "NFFT.jl/TENSOR", "NFFT.jl/POLY", "NFFT3/TENSOR", "FINUFFT", "DUCC0"]
const precomp = [NFFT.TENSOR, NFFT.POLYNOMIAL, NFFT.TENSOR, NFFT.LINEAR, NFFT.LINEAR]
const blocking = [true, true, true, true, true]

const benchmarkTime = [10, 30, 30]

ccall(("omp_set_num_threads",NFFT3.lib_path_nfft),Nothing,(Int64,),convert(Int64,Threads.nthreads()))
@info ccall(("nfft_get_num_threads",NFFT3.lib_path_nfft),Int64,())
NFFT._use_threads[] = (Threads.nthreads() > 1)


const σs = [2.0] 
const ms = 3:8
#const NBase = [65536, 256, 32] #[4*4096, 128, 32]
#const NBase = [4*4096, 128, 32]
const NBase = [512*512, 512, 64]
const Ds = 1:3
const fftflags = NFFT.FFTW.MEASURE

function nfft_accuracy_comparison(Ds=1:3)
  println("\n\n ##### nfft_performance vs accuracy ##### \n\n")

  df = DataFrame(Package=String[], D=Int[], J=Int[], N=Int[], m = Int[], σ=Float64[],
                   ErrorTrafo=Float64[], ErrorAdjoint=Float64[], 
                   TimePre=Float64[], TimeTrafo=Float64[], TimeAdjoint=Float64[],  )  

  for D in Ds
    @info "### Dimension D=$D ###"
    N = ntuple(d->NBase[D], D)
    J = prod(N)
    
    fApprox = randn(ComplexF64, N)
    gHatApprox = randn(ComplexF64, J)

    # ground truth (numerical)
    filenameCache = "./data/cache_performanceVsAccuracy_D$(D).dat"
    if !isfile(filenameCache)
      k = rand(D,J) .- 0.5
      fHat = randn(ComplexF64, J)
      #pNDFT = NDFTPlan(k, N)
      pNDFT = NFFTPlan(k, N; m=10, σ=2, precompute= NFFT.POLYNOMIAL, blocking=true, fftflags=fftflags)
      f = adjoint(pNDFT) * fHat
      gHat = pNDFT * f
      serialize(filenameCache, (k, f, fHat, gHat))
    else
      k, f, fHat, gHat = deserialize(filenameCache)
    end
    
    for σ in σs
      for m in ms
        @info "m=$m D=$D σ=$σ "

        for pl = 1:length(packagesStr)
          planner = packagesCtor[pl]
          p = planner(k, N; m, σ, precompute=precomp[pl], blocking=blocking[pl], fftflags=fftflags)
          BenchmarkTools.DEFAULT_PARAMETERS.seconds = benchmarkTime[1] 
          b = @benchmark $planner($k, $N; m=$m, σ=$σ, precompute=$(precomp[pl]), blocking=$(blocking[pl]), fftflags=$(fftflags))
          tpre = minimum(b).time / 1e9

          @info "Adjoint accuracy: $(packagesStr[pl])"
          mul!(fApprox, adjoint(p), fHat)
          eadjoint = norm(f[:] - fApprox[:],Inf) / norm(f[:],Inf)

          @info "Adjoint benchmark: $(packagesStr[pl])"
          BenchmarkTools.DEFAULT_PARAMETERS.seconds = benchmarkTime[2] 
          b = @benchmark mul!($fApprox, $(adjoint(p)), $fHat)
          tadjoint = minimum(b).time / 1e9

          @info "Trafo accuracy: $(packagesStr[pl])"
          mul!(gHatApprox, p, f)
          etrafo = norm(gHat[:] - gHatApprox[:],Inf) / norm(gHat[:],Inf)

          @info "Trafo benchmark: $(packagesStr[pl])"
          BenchmarkTools.DEFAULT_PARAMETERS.seconds = benchmarkTime[3]
          b = @benchmark mul!($gHatApprox, $p, $f)
          ttrafo = minimum(b).time / 1e9

          push!(df, (packagesStr[pl], D, J, N[D], m, σ, etrafo, eadjoint, tpre, ttrafo, tadjoint))

        end
      end
    end
  end
  return df
end



function plot_accuracy(df, packagesStr, packagesStrShort, filename)


  Plots.scalefontsizes()
  Plots.scalefontsizes(1.5)

  #colors = [RGB(0.0,0.29,0.57), RGB(0.3,0.5,0.7), RGB(0.95,0.59,0.22), RGB(1.0,0.87,0.0)]
  #colors = [RGB(0.0,0.29,0.57), RGB(0.3,0.5,0.7),  RGB(0.7,0.13,0.16), RGB(0.72,0.84,0.48)]
  #colors = [RGB(0.0,0.29,0.57), RGB(0.3,0.5,0.7), RGB(1.0,0.87,0.0), RGB(0.95,0.59,0.22)]
  #colors = [RGB(0.0,0.29,0.57), RGB(0.3,0.5,0.7), RGB(0.94,0.53,0.12), RGB(0.99,0.75,0.05)]
  colors = [RGB(0.0,0.29,0.57), RGB(0.3,0.5,0.7), RGB(0.94,0.53,0.12), RGB(0.99,0.75,0.05), RGB(1.0,0.87,0.0)]

  ls = [:solid, :solid, :solid, :solid, :solid]
  shape = [:xcross, :circle, :xcross, :cross, :cross]

  xlims = [(1e-14,1e-4), (1e-14,1e-4),(1e-14,1e-4)]
  ylims = [(0,0.053), (0,0.21),(0,2.6)]
  xticks = ([1e-14, 1e-12, 1e-10, 1e-8, 1e-6 ,1e-4],
            [L"10^{-14}", L"10^{-12}", L"10^{-10}", L"10^{-8}", L"10^{-6}" , L"10^{-4}"])
  

  pl = Matrix{Any}(undef, 3, length(Ds))
  for (i,D) in enumerate(Ds)
    titleTrafo = L"\textrm{NFFT}, \textrm{%$(D)D}"
    titleAdjoint = L"\textrm{NFFT}^H, \textrm{%$(D)D}"
    titlePre = L"\textrm{Precompute}, \textrm{%$(D)D}"
    xlabel = (i==length(Ds)) ? "Relative Error" : ""

    df1_ = df[df.σ.==2.0 .&& df.D.==D,:]  
    maxTimeTrafo = maximum( maximum(df1_[df1_.Package.==pStr,:TimeTrafo]) for pStr in packagesStr)
    maxTimeAdjoint = maximum( maximum(df1_[df1_.Package.==pStr,:TimeAdjoint]) for pStr in packagesStr)
    maxTimePre = maximum( maximum(df1_[df1_.Package.==pStr,:TimePre]) for pStr in packagesStr)

    p1 = plot(df1_[df1_.Package.==packagesStr[1],:ErrorTrafo], 
              df1_[df1_.Package.==packagesStr[1],:TimeTrafo], ylims=ylims[i],
              label = packagesStrShort[1],
              xscale = :log10, legend = (i==length(Ds)) ? (0.5, -0.5) : nothing, legend_column=length(packagesStr),
              lw=2, xlabel = xlabel, ylabel="Runtime / s",
              title=titleTrafo, shape=shape[1], ls=ls[1], 
              c=colors[1], msc=colors[1], mc=colors[1], ms=4, msw=2,
              xlims=xlims[i], xticks=xticks)

    for p=2:length(packagesStr)      
      plot!(p1, df1_[df1_.Package.==packagesStr[p],:ErrorTrafo], 
            df1_[df1_.Package.==packagesStr[p],:TimeTrafo], 
              xscale = :log10, lw=2, shape=shape[p], ls=ls[p], 
              label =  packagesStrShort[p] ,
              c=colors[p], msc=colors[p], mc=colors[p], ms=4, msw=2)
    end

    p2 = plot(df1_[df1_.Package.==packagesStr[1],:ErrorAdjoint], 
              df1_[df1_.Package.==packagesStr[1],:TimeAdjoint], ylims=ylims[i],
              xscale = :log10,  lw=2, xlabel = xlabel, #ylabel="Runtime / s", #label=packagesStr[1],
              legend = nothing, title=titleAdjoint, shape=shape[1], ls=ls[1], 
              c=colors[1], msc=colors[1], mc=colors[1], ms=4, msw=2,
              xlims=xlims[i], xticks=xticks)

    for p=2:length(packagesStr)      
      plot!(p2, df1_[df1_.Package.==packagesStr[p],:ErrorAdjoint], 
            df1_[df1_.Package.==packagesStr[p],:TimeAdjoint], 
              xscale = :log10,  lw=2, shape=shape[p], ls=ls[p], #label=packagesStr[p],
              c=colors[p], msc=colors[p], mc=colors[p], ms=4, msw=2)
    end

    p3 = plot(df1_[df1_.Package.==packagesStr[1],:ErrorAdjoint], 
              df1_[df1_.Package.==packagesStr[1],:TimePre], ylims=(0.0,maxTimePre),
              label = (i==1) ? packagesStrShort[1] : "",
              xscale = :log10,  lw=2, xlabel = xlabel, #ylabel="Runtime / s", #label=packagesStr[1],
              title=titlePre, shape=shape[1], ls=ls[1], 
              c=colors[1], msc=colors[1], mc=colors[1], ms=4, msw=2,
              legend = nothing,
              xlims=xlims[i], xticks=xticks )

    for p=2:length(packagesStr)      
      plot!(p3, df1_[df1_.Package.==packagesStr[p],:ErrorAdjoint], 
            df1_[df1_.Package.==packagesStr[p],:TimePre], 
             label = (i==1) ? packagesStrShort[p] : "",
              xscale = :log10,  lw=2, shape=shape[p], ls=ls[p], #label=packagesStr[p],
              c=colors[p], msc=colors[p], mc=colors[p], ms=4, msw=2)
    end
    pl[1,i] = p1; pl[2,i] = p2; pl[3,i] = p3; 
  end

  p = plot(vec(pl)..., layout=(length(Ds),3), size=(900,600), dpi=200, margin = 1mm, tex_output_standalone = true)
  
  mkpath("./img/")
  savefig(p, filename*".pdf")
  savefig(p, filename*".svg")
  #savefig(p, filename*".tex")
  return p
end



function plot_accuracy_small(df, packagesStr, packagesStrShort, filename)


  Plots.scalefontsizes()
  Plots.scalefontsizes(1.5)

  #colors = [RGB(0.0,0.29,0.57), RGB(0.3,0.5,0.7), RGB(0.95,0.59,0.22), RGB(1.0,0.87,0.0)]
  #colors = [RGB(0.0,0.29,0.57), RGB(0.3,0.5,0.7),  RGB(0.7,0.13,0.16), RGB(0.72,0.84,0.48)]
  #colors = [RGB(0.0,0.29,0.57), RGB(0.3,0.5,0.7), RGB(1.0,0.87,0.0), RGB(0.95,0.59,0.22)]
  colors = [ RGB(0.3,0.5,0.7), RGB(0.94,0.53,0.12), RGB(0.99,0.75,0.05), RGB(1.0,0.87,0.0)]
  ls = [:solid, :solid, :solid, :solid]
  shape = [:xcross, :circle, :xcross, :cross]

  xlims = [(4e-15,1e-5), (4e-15,1e-4),(4e-15,1e-4)]
  xticks = ([1e-14, 1e-12, 1e-10, 1e-8, 1e-6 ,1e-4],
            [L"10^{-14}", L"10^{-12}", L"10^{-10}", L"10^{-8}", L"10^{-6}" , L"10^{-4}"])
  
  pl = Matrix{Any}(undef, 3, length(Ds))
  for (i,D) in enumerate(Ds)
    titleTrafo = L"\textrm{NFFT}, \textrm{%$(D)D}"
    titleAdjoint = L"\textrm{NFFT}^H, \textrm{%$(D)D}"
    titlePre = L"\textrm{Precompute}, \textrm{%$(D)D}"
    xlabel = "Relative Error"

    df1_ = df[df.σ.==2.0 .&& df.D.==D,:]  
    maxTimeTrafo = maximum( maximum(df1_[df1_.Package.==pStr,:TimeTrafo]) for pStr in packagesStr)
    maxTimeAdjoint = maximum( maximum(df1_[df1_.Package.==pStr,:TimeAdjoint]) for pStr in packagesStr)
    maxTimePre = maximum( maximum(df1_[df1_.Package.==pStr,:TimePre]) for pStr in packagesStr)

    p1 = plot(df1_[df1_.Package.==packagesStr[1],:ErrorTrafo], 
              df1_[df1_.Package.==packagesStr[1],:TimeTrafo], ylims=(0.0,maxTimeTrafo),
              label = packagesStrShort[1],
              xscale = :log10, legend =  :topright, 
              lw=2, xlabel = xlabel, ylabel="Runtime / s",
              title=titleTrafo, shape=shape[1], ls=ls[1], 
              c=colors[1], msc=colors[1], mc=colors[1], ms=4, msw=2,
              xlims=xlims[i], xticks=xticks)

    for p=2:length(packagesStr)      
      plot!(p1, df1_[df1_.Package.==packagesStr[p],:ErrorTrafo], 
            df1_[df1_.Package.==packagesStr[p],:TimeTrafo], 
              xscale = :log10, lw=2, shape=shape[p], ls=ls[p], 
              label =  packagesStrShort[p] ,
              c=colors[p], msc=colors[p], mc=colors[p], ms=4, msw=2)
    end

    p2 = plot(df1_[df1_.Package.==packagesStr[1],:ErrorAdjoint], 
              df1_[df1_.Package.==packagesStr[1],:TimeAdjoint], ylims=(0.0,maxTimeAdjoint),
              xscale = :log10,  lw=2, xlabel = xlabel, #ylabel="Runtime / s", #label=packagesStr[1],
              legend = nothing, title=titleAdjoint, shape=shape[1], ls=ls[1], 
              c=colors[1], msc=colors[1], mc=colors[1], ms=4, msw=2,
              xlims=xlims[i], xticks=xticks)

    for p=2:length(packagesStr)      
      plot!(p2, df1_[df1_.Package.==packagesStr[p],:ErrorAdjoint], 
            df1_[df1_.Package.==packagesStr[p],:TimeAdjoint], 
              xscale = :log10,  lw=2, shape=shape[p], ls=ls[p], #label=packagesStr[p],
              c=colors[p], msc=colors[p], mc=colors[p], ms=4, msw=2)
    end

    p3 = plot(df1_[df1_.Package.==packagesStr[1],:ErrorAdjoint], 
              df1_[df1_.Package.==packagesStr[1],:TimePre], ylims=(0.0,maxTimePre),
              label = (i==1) ? packagesStrShort[1] : "",
              xscale = :log10,  lw=2, xlabel = xlabel, #ylabel="Runtime / s", #label=packagesStr[1],
              title=titlePre, shape=shape[1], ls=ls[1], 
              c=colors[1], msc=colors[1], mc=colors[1], ms=4, msw=2,
              legend = nothing,
              xlims=xlims[i], xticks=xticks )

    for p=2:length(packagesStr)      
      plot!(p3, df1_[df1_.Package.==packagesStr[p],:ErrorAdjoint], 
            df1_[df1_.Package.==packagesStr[p],:TimePre], 
             label = (i==1) ? packagesStrShort[p] : "",
              xscale = :log10,  lw=2, shape=shape[p], ls=ls[p], #label=packagesStr[p],
              c=colors[p], msc=colors[p], mc=colors[p], ms=4, msw=2)
    end
    pl[1,i] = p1; pl[2,i] = p2; pl[3,i] = p3; 
    
    p = plot(p1, p2, layout=(1,2), size=(800,300), dpi=200, margin = 1mm)

    mkpath("./img/")
    savefig(p, filename*"_$(D).pdf")
    savefig(p, filename*"_$(D).svg")
  end

end


df = nfft_accuracy_comparison(Ds)
writedlm("data/performanceVsAccuracy.csv", Iterators.flatten(([names(df)], eachrow(df))), ',')

data, header = readdlm("data/performanceVsAccuracy.csv", ',', header=true);
df = DataFrame(data, vec(header))

plot_accuracy(df, [ "NFFT.jl/POLY", "NFFT.jl/TENSOR", "NFFT3/TENSOR", "FINUFFT", "DUCC0"],
                  [ "NFFT.jl/POLY", "NFFT.jl/TENSOR", "NFFT3", "FINUFFT", "DUCC0"], "./img/performanceVsAccuracy")
                  
plot_accuracy_small(df, [ "NFFT.jl/TENSOR", "NFFT3/TENSOR", "FINUFFT", "DUCC0"],
                  [ "NFFT.jl", "NFFT3", "FINUFFT", "DUCC0"], "./img/performanceVsAccuracy")

#plot_accuracy(df, [ "NFFT.jl/POLY", "NFFT.jl/TENSOR" ], #"NFFT.jl/LINEAR"  , "LINEAR"
#                  [ "POLYNOMIAL", "TENSOR"], "./img/performanceVsAccuracyPrecomp.pdf")








