using NFFT, DataFrames, LinearAlgebra, LaTeXStrings, DelimitedFiles
using BenchmarkTools
using Plots, StatsPlots, CategoricalArrays
pgfplotsx()
#gr()



include("../Wrappers/NFFT3.jl")
include("../Wrappers/FINUFFT.jl")


const LUTSize = 20000

const threads = [1,2,4,8] #,16]
const preString = "LUT"
const preNFFTjl = [NFFT.LUT, NFFT.LUT, NFFT.TENSOR]
const packagesCtor = [NFFTPlan, FINUFFTPlan, NFFT3Plan]
const packagesStr = ["NFFT.jl","FINUFFT", "NFFT3"]
const benchmarkTime = [0.1, 4, 4]

NFFT.FFTW.set_num_threads(Threads.nthreads())
ccall(("omp_set_num_threads",NFFT3.lib_path_nfft),Nothing,(Int64,),convert(Int64,Threads.nthreads()))
@info ccall(("nfft_get_num_threads",NFFT3.lib_path_nfft),Int64,())
NFFT._use_threads[] = (Threads.nthreads() > 1)

function nfft_performance_comparison(m = 4, σ = 2.0)
  println("\n\n ##### nfft_performance_threading ##### \n\n")

  df = DataFrame(Package=String[], Threads=Int[], D=Int[], M=Int[], N=Int[], 
                   Undersampled=Bool[], Pre=String[], m = Int[], σ=Float64[],
                   TimePre=Float64[], TimeTrafo=Float64[], TimeAdjoint=Float64[] )  


  N = [collect(4096* (4 .^(0:3))),collect(128* (2 .^ (0:3))),[32,48,64,72]]
  fftflags = NFFT.FFTW.MEASURE


  for D = 2:2
    for U = 4:4
      NN = ntuple(d->N[D][U], D)
      M = prod(NN) #÷ 8

        @info D, NN, M
        
        T = Float64

        x = T.(rand(T,D,M) .- 0.5)
        x .= sortslices(x, dims=2) # sort nodes to gain cache locality
        
        fHat = randn(Complex{T}, M)
        f = randn(Complex{T}, NN)
        
        for pl = 1:length(packagesStr)

          planner = packagesCtor[pl]
          BenchmarkTools.DEFAULT_PARAMETERS.seconds = benchmarkTime[1]
          b = @benchmark $planner($x, $NN; m=$m, σ=$σ, window=:kaiser_bessel, LUTSize=$LUTSize, 
                                 precompute=$(preNFFTjl[pl]), sortNodes=false, fftflags=$fftflags)
          tpre = minimum(b).time / 1e9
          p = planner(x, NN; m=m, σ=σ, window=:kaiser_bessel, LUTSize=LUTSize, 
                      precompute=(preNFFTjl[pl]), sortNodes=false, fftflags=fftflags)
          BenchmarkTools.DEFAULT_PARAMETERS.seconds = benchmarkTime[2]                      
          b = @benchmark mul!($f, $(adjoint(p)), $fHat)
          tadjoint = minimum(b).time / 1e9
          BenchmarkTools.DEFAULT_PARAMETERS.seconds = benchmarkTime[3]
          b = @benchmark mul!($fHat, $p, $f)
          ttrafo = minimum(b).time / 1e9
        
          push!(df, (packagesStr[pl], Threads.nthreads(), D, M, N[D][U], false, preString, m, σ,
                   tpre, ttrafo, tadjoint))

      end
    end
  end
  return df
end



function plot_performance(df; pre = "LUT")

  Plots.scalefontsizes()
  Plots.scalefontsizes(1.5)

  labelsA = packagesStr

  tpre = zeros(length(threads),length(labelsA))
  ttrafo = zeros(length(threads), length(labelsA))
  tadjoint = zeros(length(threads), length(labelsA))
  for (i,p) in enumerate(packagesStr)
    for (j,th) in enumerate(threads)
      tpre[j,i] = df[df.Threads .== th .&& df.Package.==p .&& df.Pre.==pre,:TimePre][1]
      ttrafo[j,i] = df[df.Threads .== th .&& df.Package.==p .&& df.Pre.==pre,:TimeTrafo][1]
      tadjoint[j,i] = df[df.Threads .== th .&& df.Package.==p .&& df.Pre.==pre,:TimeAdjoint][1]
    end
  end
  
  
  labelsB = "t = " .* string.(threads) 

  
  ctg = CategoricalArray(repeat(labelsA, inner = length(threads)))
  levels!(ctg, labelsA)
  name = CategoricalArray(repeat(labelsB, outer = length(labelsA)))
  levels!(name, labelsB)
  
  p1 = groupedbar(name, tpre, ylabel = "time / s",  group = ctg,
          bar_width = 0.67,
          lw = 0, framestyle = :box, size=(800,600), title = L"\textrm{Precompute}")

  p2 = groupedbar(name, ttrafo, ylabel = "time / s",  group = ctg,
          bar_width = 0.67, legend = :none,
          lw = 0, framestyle = :box, size=(800,600), title = L"\textrm{NFFT}")

  p3 = groupedbar(name, tadjoint, ylabel = "time / s",  group = ctg,
          bar_width = 0.67, legend = :none,
          lw = 0, framestyle = :box, size=(800,600), title = L"\textrm{NFFT}^H")
  
  p = plot(p1, p2, p3, layout=(3,1), size=(800,600), dpi=200)

  savefig(p, "../docs/src/assets/performance_mt_$(pre).svg")
  return p
end



function plot_performance_serial(df)

  Plots.scalefontsizes()
  Plots.scalefontsizes(1.5)

  tNFFTjl = zeros(2,3)
  tNFFT3 = zeros(2,3)

  for (j,pre) in enumerate(["LUT", "FULL"])
    tNFFTjl[j,1] = df[df.Package.=="NFFT.jl" .&& df.Pre.==pre .&& df.Threads.==1,:TimePre][1]
    tNFFTjl[j,2] = df[df.Package.=="NFFT.jl" .&& df.Pre.==pre .&& df.Threads.==1,:TimeTrafo][1]
    tNFFTjl[j,3] = df[df.Package.=="NFFT.jl" .&& df.Pre.==pre .&& df.Threads.==1,:TimeAdjoint][1]
    
    tNFFT3[j,1] = df[df.Package.=="NFFT3" .&& df.Pre.==pre .&& df.Threads.==1,:TimePre ][1]
    tNFFT3[j,2] = df[df.Package.=="NFFT3" .&& df.Pre.==pre .&& df.Threads.==1,:TimeTrafo][1]
    tNFFT3[j,3] = df[df.Package.=="NFFT3" .&& df.Pre.==pre .&& df.Threads.==1,:TimeAdjoint][1]
  end
   
  labelsA = ["Precompute", "NFFT", "adjoint NFFT"]
  labelsB = [L"\textrm{LUT}", L"\textrm{FULL}"]

  
  ctg = CategoricalArray(repeat(labelsA, inner = 2))
  levels!(ctg, labelsA)
  name = CategoricalArray(repeat(labelsB, outer = 3))
  levels!(name, labelsB)
  
  tmax = max(maximum(tNFFTjl),maximum(tNFFT3))
  
  p1 = groupedbar(name, tNFFTjl, ylabel = "time / s",  group = ctg,
          bar_width = 0.67, title = "NFFT.jl", legend = :none,
          lw = 0,  size=(800,600), framestyle = :box, ylim=(0,tmax))
          
  p2 = groupedbar(name, tNFFT3, ylabel = "time / s",  group = ctg,
          bar_width = 0.67, title = "NFFT3",
          lw = 0,  size=(800,600), framestyle = :box, ylim=(0,tmax))
  
  p = plot(p1, p2, layout=(1,2), size=(800,600), dpi=200)
  
  savefig(p, "../docs/src/assets/performance_serial.svg")
  return p
end




### run the code ###

if haskey(ENV, "NFFT_PERF_THREADING")
  df = nfft_performance_comparison(4, 2.0)

  if isfile("performance_mt.csv")
    data, header = readdlm("performance_mt.csv", ',', header=true);
    df_ = DataFrame(data, vec(header))
    append!(df, df_)
  end

  writedlm("performance_mt.csv", Iterators.flatten(([names(df)], eachrow(df))), ',')

else
  rm("performance_mt.csv", force=true)
  ENV["NFFT_PERF_THREADING"] = 1
  for t in threads
    cmd = `julia -t $t performance_threading.jl`
    @info cmd
    run(cmd)

  end
  data, header = readdlm("performance_mt.csv", ',', header=true);
  df = DataFrame(data, vec(header))
  delete!(ENV, "NFFT_PERF_THREADING")

  plot_performance(df, pre="LUT")
  #plot_performance(df, pre="FULL") 
  #plot_performance_serial(df)
end
