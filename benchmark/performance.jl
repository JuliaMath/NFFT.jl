using NFFT, DataFrames, LinearAlgebra, LaTeXStrings, DelimitedFiles
using BenchmarkTools
using Plots, StatsPlots 
pgfplotsx()
#gr()

BenchmarkTools.DEFAULT_PARAMETERS.seconds = 5

include("../NFFT3/NFFT3.jl")


const LUTSize = 20000

NFFT.FFTW.set_num_threads(Threads.nthreads())
ccall(("omp_set_num_threads",NFFT3.lib_path_nfft),Nothing,(Int64,),convert(Int64,Threads.nthreads()))
@info ccall(("nfft_get_num_threads",NFFT3.lib_path_nfft),Int64,())
if Threads.nthreads() == 1
  NFFT._use_threads[] = false
end

function nfft_performance_comparison(m = 6, σ = 2.0)
  println("\n\n ##### nfft_performance_comparison ##### \n\n")

  df = DataFrame(Package=String[], D=Int[], M=Int[], N=Int[], 
                   Undersampled=Bool[], Pre=String[], m = Int[], σ=Float64[],
                   TimePre=Float64[], TimeTrafo=Float64[], TimeAdjoint=Float64[] )  

  preString = ["LUT", "FULL", "FULL_LUT"]
  preNFFTjl = [NFFT.LUT, NFFT.FULL, NFFT.FULL_LUT]
  N = [collect(4096* (4 .^(0:3))),collect(128* (2 .^ (0:3))),[32,48,64,72]]
  fftflags = NFFT.FFTW.MEASURE

  for D = 2:2
    for U = 4:4
      NN = ntuple(d->N[D][U], D)
      M = prod(NN) ÷ 10 #*2 #

      for pre = 1:2

        @info D, NN, M, pre
        
        T = Float64

        x = T.(rand(T,D,M) .- 0.5)
        fHat = randn(Complex{T}, M)
        fApprox = randn(Complex{T}, NN)

        tpre = @belapsed plan_nfft($x, $NN; m=$m, σ=$σ, window=:kaiser_bessel, LUTSize=$LUTSize, precompute=$(preNFFTjl[pre]), sortNodes=false, fftflags=$fftflags)
        p = plan_nfft(x, NN; m=m, σ=σ, window=:kaiser_bessel, LUTSize=LUTSize, precompute=(preNFFTjl[pre]), sortNodes=false, fftflags=fftflags)
        f = similar(fHat, p.N)
        tadjoint = @belapsed nfft_adjoint!($p, $fHat, $f)
        ttrafo = @belapsed nfft!($p, $fApprox, $fHat)
        
        push!(df, ("NFFT.jl", D, M, N[D][U], false, preString[pre], m, σ,
                   tpre, ttrafo, tadjoint))

        tpre = @belapsed NFFT3Plan($x, $NN; m=$m, σ=$σ, window=:kaiser_bessel, LUTSize=$LUTSize, precompute=$(preNFFTjl[pre]), sortNodes=true, fftflags=$fftflags)
        p = NFFT3Plan(x, NN; m, σ, window=:kaiser_bessel, LUTSize, precompute=preNFFTjl[pre], sortNodes=true, fftflags=fftflags)
        tadjoint = @belapsed nfft_adjoint!($p, $fHat, $f)
        ttrafo = @belapsed nfft!($p, $fApprox, $fHat)
                
        push!(df, ("NFFT3", D, M, N[D][U], false, preString[pre], m, σ,
                    tpre, ttrafo, tadjoint))
          
      end
    end
  end
  return df
end

#df = nfft_performance_comparison(4, 2.0)
#writedlm("performance.csv", Iterators.flatten(([names(df)], eachrow(df))), ',')

data, header = readdlm("performance.csv", ',', header=true);
df = DataFrame(data, vec(header))

function plot_performance_1(df)

  Plots.scalefontsizes()
  Plots.scalefontsizes(1.5)

  t = zeros(4,3)
  for (i,p) in enumerate(["NFFT.jl", "NFFT3"])
    for (j,pre) in enumerate(["LUT", "FULL"])
      t[2*(j-1)+i,1] = df[df.Package.==p .&& df.Pre.==pre,:TimePre][1]
      t[2*(j-1)+i,2] = df[df.Package.==p .&& df.Pre.==pre,:TimeTrafo][1]
      t[2*(j-1)+i,3] = df[df.Package.==p .&& df.Pre.==pre,:TimeAdjoint][1]
    end
  end
  labels = ["Pre", "NFFT", L"\textrm{NFFT}^H"]

  ctg = repeat(["Pre", "NFFT", L"\textrm{NFFT}^H"], inner = 4)
  nam = repeat( [L"\textrm{NFFT.jl / LUT}", L"\textrm{NFFT3 / LUT}", 
                 L"\textrm{NFFT.jl / FULL}", L"\textrm{NFFT3 / FULL}"] , outer = 3)
  
  groupedbar(nam, t, ylabel = "time / s",  #group = ctg,
          bar_width = 0.67,
          lw = 0, framestyle = :box, size=(800,600))
  

end

plot_performance_1(df)
