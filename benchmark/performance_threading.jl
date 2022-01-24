using NFFT, DataFrames, LinearAlgebra, LaTeXStrings, DelimitedFiles
using BenchmarkTools
using Plots, StatsPlots, CategoricalArrays
pgfplotsx()
#gr()

BenchmarkTools.DEFAULT_PARAMETERS.seconds = 40

include("../NFFT3/NFFT3.jl")


const LUTSize = 20000

threads = [1,2,4,8,16]

NFFT.FFTW.set_num_threads(Threads.nthreads())
ccall(("omp_set_num_threads",NFFT3.lib_path_nfft),Nothing,(Int64,),convert(Int64,Threads.nthreads()))
@info ccall(("nfft_get_num_threads",NFFT3.lib_path_nfft),Int64,())
NFFT._use_threads[] = (Threads.nthreads() > 1)

function nfft_performance_comparison(m = 6, σ = 2.0)
  println("\n\n ##### nfft_performance_threading ##### \n\n")

  df = DataFrame(Package=String[], Threads=Int[], D=Int[], M=Int[], N=Int[], 
                   Undersampled=Bool[], Pre=String[], m = Int[], σ=Float64[],
                   TimePre=Float64[], TimeTrafo=Float64[], TimeAdjoint=Float64[] )  

  preString = ["LUT", "FULL", "FULL_LUT"]
  preNFFTjl = [NFFT.LUT, NFFT.FULL, NFFT.FULL_LUT]
  N = [collect(4096* (4 .^(0:3))),collect(128* (2 .^ (0:3))),[32,48,64,72]]
  fftflags = NFFT.FFTW.MEASURE

  for D = 2:2
    for U = 4:4
      NN = ntuple(d->N[D][U], D)
      M = prod(NN) #÷ 8

      for pre = 1:2

        @info D, NN, M, pre
        
        T = Float64

        x = T.(rand(T,D,M) .- 0.5)
        x .= sortslices(x, dims=2) # sort nodes to gain cache locality
        
        fHat = randn(Complex{T}, M)
        fApprox = randn(Complex{T}, NN)

        tpre = @belapsed plan_nfft($x, $NN; m=$m, σ=$σ, window=:kaiser_bessel, LUTSize=$LUTSize, precompute=$(preNFFTjl[pre]), sortNodes=false, fftflags=$fftflags)
        p = plan_nfft(x, NN; m=m, σ=σ, window=:kaiser_bessel, LUTSize=LUTSize, precompute=(preNFFTjl[pre]), sortNodes=false, fftflags=fftflags)
        f = similar(fHat, p.N)
        tadjoint = @belapsed nfft_adjoint!($p, $fHat, $f)
        ttrafo = @belapsed nfft!($p, $fApprox, $fHat)
        
        push!(df, ("NFFT.jl", Threads.nthreads(), D, M, N[D][U], false, preString[pre], m, σ,
                   tpre, ttrafo, tadjoint))

        tpre = @belapsed NFFT3Plan($x, $NN; m=$m, σ=$σ, window=:kaiser_bessel, LUTSize=$LUTSize, precompute=$(preNFFTjl[pre]), sortNodes=true, fftflags=$fftflags)
        p = NFFT3Plan(x, NN; m, σ, window=:kaiser_bessel, LUTSize, precompute=preNFFTjl[pre], sortNodes=true, fftflags=fftflags)
        tadjoint = @belapsed nfft_adjoint!($p, $fHat, $f)
        ttrafo = @belapsed nfft!($p, $fApprox, $fHat)
                
        push!(df, ("NFFT3", Threads.nthreads(), D, M, N[D][U], false, preString[pre], m, σ,
                    tpre, ttrafo, tadjoint))
          
      end
    end
  end
  return df
end



function plot_performance(df; pre = "FULL")

  Plots.scalefontsizes()
  Plots.scalefontsizes(1.5)

  tpre = zeros(length(threads),2)
  ttrafo = zeros(length(threads),2)
  tadjoint = zeros(length(threads),2)
  for (i,p) in enumerate(["NFFT.jl", "NFFT3"])
    for (j,th) in enumerate(threads)
      tpre[j,i] = df[df.Threads .== th .&& df.Package.==p .&& df.Pre.==pre,:TimePre][1]
      ttrafo[j,i] = df[df.Threads .== th .&& df.Package.==p .&& df.Pre.==pre,:TimeTrafo][1]
      tadjoint[j,i] = df[df.Threads .== th .&& df.Package.==p .&& df.Pre.==pre,:TimeAdjoint][1]
    end
  end
  
  labelsA = ["NFFT.jl", "NFFT3"]
  labelsB = "t = " .* string.(threads) 

  
  ctg = CategoricalArray(repeat(labelsA, inner = length(threads)))
  levels!(ctg, labelsA)
  name = CategoricalArray(repeat(labelsB, outer = 2))
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
  plot_performance(df, pre="FULL")  
end
