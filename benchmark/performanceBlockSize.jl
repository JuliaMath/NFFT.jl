using NFFT, DataFrames, LinearAlgebra, LaTeXStrings, DelimitedFiles
using BenchmarkTools
using Plots; pgfplotsx()

const benchmarkTime = [20, 20]

NFFT.FFTW.set_num_threads(Threads.nthreads())
NFFT._use_threads[] = (Threads.nthreads() > 1)

const threads = [1,2,4,8] 

#const NBase = [65536, 256, 32] 
const NBase = [512*512, 512, 64]
const Ds = 1:3
const blockSizeBase = [
  [nextpow(2,round(Int,x)) for x=2 .^range(1,19,length=7)], #[div(NBase[1]*2,2^j) for j=(0:6).*2],
  [nextpow(2,round(Int,x)) for x=2 .^range(1,10,length=7)], #[div(NBase[2]*2,2^j) for j=0:6],
  [nextpow(2,round(Int,x)) for x=2 .^range(1,7,length=7)] #,[div(NBase[3]*2,2^j) for j=0:6],
]


const fftflags = NFFT.FFTW.MEASURE

function nfft_block_size_comparison(Ds=1:3, m = 4, σ = 2.0)
  println("\n\n ##### nfft_block_size_comparison ##### \n\n")

  df = DataFrame(Package=String[], Threads=Int[], D=Int[], J=Int[], N=Int[], m = Int[], σ=Float64[],
                   TimeTrafo=Float64[], TimeAdjoint=Float64[], blocking=Bool[], blockSizeBase=Int[] )  

  for D in Ds
    @info "### Dimension D=$D ###"
    N = ntuple(d->NBase[D], D)
    J = prod(N)
    
    k = rand(D,J) .- 0.5
    fHat = randn(ComplexF64, J)
    fApprox = randn(ComplexF64, N)
    gHatApprox = randn(ComplexF64, J)

    for bl in blockSizeBase[D]
        @info "b=$(bl) D=$D"

          blockSize = ntuple(d-> bl, D)
          p = NFFTPlan(k, N; m, σ, precompute=POLYNOMIAL, blocking=true, blockSize=blockSize, fftflags=fftflags)

          @info "Adjoint benchmark:"
          BenchmarkTools.DEFAULT_PARAMETERS.seconds = benchmarkTime[1] 
          b = @benchmark mul!($fApprox, $(adjoint(p)), $fHat)
          tadjoint = minimum(b).time / 1e9

          @info "Trafo benchmark:"
          BenchmarkTools.DEFAULT_PARAMETERS.seconds = benchmarkTime[2]
          b = @benchmark mul!($gHatApprox, $p, $fApprox)
          ttrafo = minimum(b).time / 1e9

          push!(df, ("NFFT.jl", Threads.nthreads(), D, J, N[D], m, σ, ttrafo, tadjoint, true, bl))
    end

    # non blocking
    p = NFFTPlan(k, N; m, σ, precompute=POLYNOMIAL, blocking=false)

    @info "Adjoint benchmark:"
    BenchmarkTools.DEFAULT_PARAMETERS.seconds = benchmarkTime[1] 
    b = @benchmark mul!($fApprox, $(adjoint(p)), $fHat)
    tadjoint = minimum(b).time / 1e9

    @info "Trafo benchmark:"
    BenchmarkTools.DEFAULT_PARAMETERS.seconds = benchmarkTime[2]
    b = @benchmark mul!($gHatApprox, $p, $fApprox)
    ttrafo = minimum(b).time / 1e9

    push!(df, ("NFFT.jl", Threads.nthreads(), D, J, N[D], m, σ, ttrafo, tadjoint, false, 0))
  end
  return df
end



function plot_performance_block_size(df, Ds)

  Plots.scalefontsizes()
  Plots.scalefontsizes(1.5)
  

  #colors = [RGB(0.0,0.29,0.57), RGB(0.3,0.5,0.7), RGB(0.95,0.59,0.22), RGB(1.0,0.87,0.0)]
  #colors = [RGB(0.0,0.29,0.57), RGB(0.3,0.5,0.7),  RGB(0.7,0.13,0.16), RGB(0.72,0.84,0.48)]
  #colors = [RGB(0.0,0.29,0.57), RGB(0.3,0.5,0.7), RGB(1.0,0.87,0.0), RGB(0.95,0.59,0.22)]
  colors = [RGB(0.72,0.84,0.48), RGB(0.41,0.76,0.80), RGB(0.5,0.48,0.45), RGB(0.7,0.13,0.16)]
  ls = [:solid, :solid, :solid, :solid]
  shape = [:xcross, :circle, :xcross, :cross]

  pl = Matrix{Any}(undef, 2, length(Ds))
  for (i,D) in enumerate(Ds)
    titleTrafo = L"\textrm{NFFT}, \textrm{%$(D)D}"
    titleAdjoint = L"\textrm{NFFT}^H, \textrm{%$(D)D}"

    df1_ = df[df.σ.==2.0 .&& df.D.==D .&& df.blocking .== true ,:]
    df2_ = df[df.σ.==2.0 .&& df.D.==D .&& df.blocking .== false ,:]
    maxTimeTrafo = max(maximum(df1_[:,:TimeTrafo]), maximum(df2_[:,:TimeTrafo]))*1.1
    maxTimeAdjoint = max(maximum(df1_[:,:TimeAdjoint]),maximum(df2_[:,:TimeAdjoint]))*1.1

    blockSizes = df1_[df1_.Threads .== threads[1], :blockSizeBase]

    p1 = plot(blockSizes, yscale = :log2, xscale = :log10,
              df1_[df1_.Threads .== threads[1], :TimeTrafo], #ylims=(0.0,maxTimeTrafo),
              label="1 thread", lw=2, ylabel="Runtime / s", xlabel = i==length(Ds) ? "Block Size" : "",
              legend = nothing, title=titleTrafo, 
              shape=shape[1], ls=ls[1], 
              c=colors[1], msc=colors[1], mc=colors[1], ms=4, msw=2)

    for p=2:length(threads)      
      plot!(p1, blockSizes, 
            df1_[df1_.Threads .== threads[p],:TimeTrafo], 
              label="$(threads[p]) threads", lw=2, shape=shape[p], ls=ls[p], 
              c=colors[p], msc=colors[p], mc=colors[p], ms=4, msw=2)
    end
    

    plot!(p1, blockSizes, 
              df2_[df2_.Threads .== threads[1], :TimeTrafo].*ones(length(blockSizes)), #ylims=(0.0,maxTimeTrafo),
              label="regular: 1 thread", lw=2,  ls=:dash,
              c=colors[1])

    for p=2:length(threads)      
      plot!(p1, blockSizes, 
              df2_[df2_.Threads .== threads[p], :TimeTrafo].*ones(length(blockSizes)), 
              label="regular: $(threads[p]) threads", lw=2,  ls=:dash, 
              c=colors[p])
    end

    p2 = plot(blockSizes, yscale = :log2, xscale = :log10,
              df1_[df1_.Threads .== threads[1],:TimeAdjoint], #ylims=(0.0,maxTimeAdjoint),
              lw=2, xlabel = i==length(Ds) ? "Block Size" : "", label="1 thread",
              legend = i==2 ? :topright : nothing, title=titleAdjoint, 
              shape=shape[1], ls=ls[1], 
              c=colors[1], msc=colors[1], mc=colors[1], ms=4, msw=2)

    for p=2:length(threads)      
      plot!(p2, blockSizes, 
            df1_[df1_.Threads .== threads[p],:TimeAdjoint], 
              label="$(threads[p]) threads", lw=2, shape=shape[p], ls=ls[p], 
              c=colors[p], msc=colors[p], mc=colors[p], ms=4, msw=2)
    end

    plot!(p2, blockSizes, 
              df2_[df2_.Threads .== threads[1], :TimeAdjoint].*ones(length(blockSizes)), #ylims=(0.0,maxTimeAdjoint),
              lw=2, ls=:dash, label=nothing,
              c=colors[1])
    pl[1,i] = p1; pl[2,i] = p2;
  end

  #p = plot(p1, p2, layout=(1,2), size=(800,300), dpi=200)
  p = plot(vec(pl)..., layout=(length(Ds),2), size=(800,700), dpi=200)

  mkpath("./img/")
  savefig(p, "./img/performanceBlockSize.pdf")
  return p
end






### run the code ###

if haskey(ENV, "NFFT_PERF")
  df = nfft_block_size_comparison(Ds)

  if isfile("./data/performanceBlockSize.csv")
    data, header = readdlm("./data/performanceBlockSize.csv", ',', header=true);
    df_ = DataFrame(data, vec(header))
    append!(df, df_)
  end

  writedlm("./data/performanceBlockSize.csv", Iterators.flatten(([names(df)], eachrow(df))), ',')

else
  if true
    rm("./data/performanceBlockSize.csv", force=true)
    ENV["NFFT_PERF"] = 1
    for t in threads
      cmd = `julia -t $t performanceBlockSize.jl`
      @info cmd
      run(cmd)
    end

    delete!(ENV, "NFFT_PERF")
  end

  data, header = readdlm("./data/performanceBlockSize.csv", ',', header=true);
  df = DataFrame(data, vec(header))

  plot_performance_block_size(df, Ds)
end









