using NFFT, DataFrames, LinearAlgebra, LaTeXStrings, DelimitedFiles
using BenchmarkTools
using Plots, StatsPlots, CategoricalArrays
pgfplotsx()
#gr()


include("../../Wrappers/NFFT3.jl")
include("../../Wrappers/FINUFFT.jl")
mkpath("./img/")
mkpath("./data/")


const threads = [1,2,4,8]
const precomp = [NFFT.POLYNOMIAL, NFFT.TENSOR, NFFT.POLYNOMIAL, NFFT.TENSOR]
const packagesCtor = [NFFTPlan, NFFTPlan, FINUFFTPlan, NFFT3Plan]
const packagesStr = ["NFFT.jl/POLY", "NFFT.jl/TENSOR", "FINUFFT", "NFFT3"]
const benchmarkTime = [1, 120, 120]
#const benchmarkTime = [1, 2, 2]
#const NBase = [4*4096, 256, 32]
const NBase = [512*512, 512, 64]

ccall(("omp_set_num_threads",NFFT3.lib_path_nfft),Nothing,(Int64,),convert(Int64,Threads.nthreads()))
@info ccall(("nfft_get_num_threads",NFFT3.lib_path_nfft),Int64,())
NFFT._use_threads[] = (Threads.nthreads() > 1)

function nfft_performance_comparison(m = 4, σ = 2.0)
  println("\n\n ##### nfft_performance ##### \n\n")

  df = DataFrame(Package=String[], Threads=Int[], D=Int[], J=Int[], N=Int[], 
                   Undersampled=Bool[], m = Int[], σ=Float64[],
                   TimePre=Float64[], TimeTrafo=Float64[], TimeAdjoint=Float64[] )  

  fftflags = NFFT.FFTW.MEASURE


  for D = 2:2
      NN = ntuple(d->NBase[D], D)
      J = prod(NN) #÷ 8

        @info D, NN, J
        
        T = Float64

        k =T.(rand(T,D,J) .- 0.5)
        
        fHat = randn(Complex{T}, J)
        f = randn(Complex{T}, NN)
        
        for pl = 1:length(packagesStr)
          @info packagesStr[pl]
          
          planner = packagesCtor[pl]
          BenchmarkTools.DEFAULT_PARAMETERS.seconds = benchmarkTime[1]
          b = @benchmark $planner($k, $NN; m=$m, σ=$σ, window=:kaiser_bessel, 
                                 precompute=$(precomp[pl]), sortNodes=false, fftflags=$fftflags)
          tpre = minimum(b).time / 1e9
          p = planner(k, NN; m=m, σ=σ, window=:kaiser_bessel, 
                      precompute=(precomp[pl]), sortNodes=false, fftflags=fftflags)
          BenchmarkTools.DEFAULT_PARAMETERS.seconds = benchmarkTime[2]                      
          b = @benchmark mul!($f, $(adjoint(p)), $fHat)
          tadjoint = minimum(b).time / 1e9
          BenchmarkTools.DEFAULT_PARAMETERS.seconds = benchmarkTime[3]
          b = @benchmark mul!($fHat, $p, $f)
          ttrafo = minimum(b).time / 1e9      

          push!(df, (packagesStr[pl], Threads.nthreads(), D, J, NBase[D], false, m, σ,
                   tpre, ttrafo, tadjoint))

    end
  end
  return df
end



function plot_performance(df; D=2, N=1024, J=N*N)

  Plots.scalefontsizes()
  Plots.scalefontsizes(1.5)

  labelsA = packagesStr

  tpre = zeros(length(threads),length(labelsA))
  ttrafo = zeros(length(threads), length(labelsA))
  tadjoint = zeros(length(threads), length(labelsA))
  for (i,p) in enumerate(packagesStr)
    for (j,th) in enumerate(threads) #.&& df.Pre.==precomp[i] 
      df_ = df[df.Threads .== th .&& df.Package.==p .&& df.D .== D .&& df.J.== J .&& df.N .==N ,:]
      tpre[j,i] = df_[1,:TimePre]
      ttrafo[j,i] = df_[1,:TimeTrafo]
      tadjoint[j,i] = df_[1,:TimeAdjoint]
    end
  end
  
  
  labelsB = "t = " .* string.(threads) 

  
  ctg = CategoricalArray(repeat(labelsA, inner = length(threads)))
  levels!(ctg, labelsA)
  name = CategoricalArray(repeat(labelsB, outer = length(labelsA)))
  levels!(name, labelsB)
  

  maxtime = max(maximum(tpre), maximum(ttrafo), maximum(tadjoint))

  p1 = groupedbar(name, ttrafo, ylabel = "time / s",  group = ctg,
          bar_width = 0.67, legend = :none, ylims=(0,maxtime),
          lw = 0, framestyle = :box, size=(800,600), title = L"\textrm{NFFT}")

  p2 = groupedbar(name, tadjoint, ylabel = "time / s",  group = ctg,
          bar_width = 0.67, legend = :none, ylims=(0,maxtime),
          lw = 0, framestyle = :box, size=(800,600), title = L"\textrm{NFFT}^H")
          
  p3 = groupedbar(name, tpre, ylabel = "time / s",  group = ctg,
          bar_width = 0.67, ylims=(0,maxtime),
          lw = 0, framestyle = :box, size=(800,600), title = L"\textrm{Precompute}")
  
  p = plot(p1, p2, p3, layout=(3,1), size=(800,600), dpi=200, tex_output_standalone = true)

  savefig(p, "./img/performance_mt.pdf")
  savefig(p, "./img/performance_mt.tex")
  return p
end



function plot_performance_speedup(df, packagesStr, packagesStrShort, colors; D=2, N=1024, J=N*N)

  Plots.scalefontsizes()
  Plots.scalefontsizes(1.5)

  labelsA = packagesStr

  tpre = zeros(length(threads),length(labelsA))
  ttrafo = zeros(length(threads), length(labelsA))
  tadjoint = zeros(length(threads), length(labelsA))
  for (i,p) in enumerate(packagesStr)
    for (j,th) in enumerate(threads) #.&& df.Pre.==precomp[i] 
      df_ = df[df.Threads .== th .&& df.Package.==p .&& df.D .== D .&& df.J.== J .&& df.N .== N ,:]
      tpre[j,i] = df_[1,:TimePre]
      ttrafo[j,i] = df_[1,:TimeTrafo]
      tadjoint[j,i] = df_[1,:TimeAdjoint]
    end
  end
  
  efftrafo = 1 ./ ttrafo .* ttrafo[1:1,:] ./ threads
  effadjoint = 1 ./ tadjoint .* tadjoint[1:1,:] ./ threads


  Plots.scalefontsizes()
  Plots.scalefontsizes(1.5)
  

  ls = [:solid, :solid, :solid, :solid]
  shape = [:xcross, :circle, :xcross, :cross]


    titleTrafo = L"\textrm{NFFT}"
    titleAdjoint = L"\textrm{NFFT}^H"

    p1 = plot(threads, 
              ttrafo[:,1], ylims=(0,0.135),
              label=packagesStrShort[1], lw=2, ylabel="Runtime / s", #xlabel = "# threads",
              legend = :topright, title=titleTrafo, 
              shape=shape[1], ls=ls[1], 
              c=colors[1], msc=colors[1], mc=colors[1], ms=4, msw=2)

    for p=2:length(packagesStr)      
      plot!(p1, threads, 
             ttrafo[:,p], 
              label=packagesStrShort[p], lw=2, shape=shape[p], ls=ls[p], 
              c=colors[p], msc=colors[p], mc=colors[p], ms=4, msw=2)
    end

    p2 = plot(threads, tadjoint[:,1], ylims=(0,0.135),
              lw=2, #xlabel = "# threads", 
              label=packagesStr[1],
              #legend = i==2 ? :topright : nothing, 
              legend = nothing,
              title=titleAdjoint, 
              shape=shape[1], ls=ls[1], 
              c=colors[1], msc=colors[1], mc=colors[1], ms=4, msw=2)

    for p=2:length(packagesStr)      
      plot!(p2, threads, tadjoint[:,p], 
              label=packagesStrShort[p], lw=2, shape=shape[p], ls=ls[p], 
              c=colors[p], msc=colors[p], mc=colors[p], ms=4, msw=2)
    end

  
    p3 = plot(threads, 
              efftrafo[:,1], ylims=(0.0,1.1),
              label=packagesStrShort[1], lw=2, ylabel="Efficiency", xlabel = "# threads",
              legend = nothing, title=titleTrafo, 
              shape=shape[1], ls=ls[1], 
              c=colors[1], msc=colors[1], mc=colors[1], ms=4, msw=2)

    for p=2:length(packagesStr)      
      plot!(p3, threads, 
             efftrafo[:,p], 
              label=packagesStrShort[p], lw=2, shape=shape[p], ls=ls[p], 
              c=colors[p], msc=colors[p], mc=colors[p], ms=4, msw=2)
    end

    p4 = plot(threads, effadjoint[:,1],  ylims=(0.0,1.1),
              lw=2, xlabel = "# threads", label=packagesStr[1],
              #legend = i==2 ? :topright : nothing, 
              legend = nothing,
              title=titleAdjoint, 
              shape=shape[1], ls=ls[1], 
              c=colors[1], msc=colors[1], mc=colors[1], ms=4, msw=2)

    for p=2:length(packagesStr)      
      plot!(p4, threads, effadjoint[:,p], 
              label=packagesStrShort[p], lw=2, shape=shape[p], ls=ls[p], 
              c=colors[p], msc=colors[p], mc=colors[p], ms=4, msw=2)
    end

   p = plot(p1, p2, p3, p4, layout=(2,2), size=(800,500), dpi=200, tex_output_standalone = true)


  savefig(p, "./img/performance_mt_speedup.pdf")
  savefig(p, "./img/performance_mt_speedup.tex")
  
  return 
end



### run the code ###

if haskey(ENV, "NFFT_PERF")
  df = nfft_performance_comparison(4, 2.0)

  if isfile("./data/performance_mt.csv")
    data, header = readdlm("./data/performance_mt.csv", ',', header=true);
    df_ = DataFrame(data, vec(header))
    append!(df, df_)
  end

  writedlm("./data/performance_mt.csv", Iterators.flatten(([names(df)], eachrow(df))), ',')

else
  if false
    rm("./data/performance_mt.csv", force=true)
    ENV["NFFT_PERF"] = 1
    for t in threads
      cmd = `julia -t $t performanceThreading.jl`
      @info cmd
      run(cmd)

    end
    delete!(ENV, "NFFT_PERF")
  end
  data, header = readdlm("./data/performance_mt.csv", ',', header=true);
  df = DataFrame(data, vec(header))

  plot_performance(df, N=NBase[2], J=NBase[2]*NBase[2])
  plot_performance_speedup(df, [ "NFFT.jl/POLY", "NFFT.jl/TENSOR", "NFFT3", "FINUFFT"],
                               [ "NFFT.jl/POLY", "NFFT.jl/TENSOR", "NFFT3", "FINUFFT"],
                               [RGB(0.0,0.29,0.57), RGB(0.3,0.5,0.7), RGB(0.94,0.53,0.12), RGB(0.99,0.75,0.05)],
                               N=NBase[2], J=NBase[2]*NBase[2])
                              
  #plot_performance_speedup(df, [ "NFFT.jl/TENSOR", "NFFT3", "FINUFFT"],
  #                             [ "NFFT.jl", "NFFT3", "FINUFFT"],
  #                             [RGB(0.3,0.5,0.7), RGB(0.94,0.53,0.12), RGB(0.99,0.75,0.05)],
  #                             N=NBase[2], J=NBase[2]*NBase[2])
end


















