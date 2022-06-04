using NFFT, DataFrames, LinearAlgebra, LaTeXStrings, DelimitedFiles
using BenchmarkTools
using Plots; pgfplotsx()

const benchmarkTime = [2, 2]

NFFT.FFTW.set_num_threads(Threads.nthreads())
NFFT._use_threads[] = (Threads.nthreads() > 1)

const threads = [1,2,4] #,8]

#const NBase = [65536, 256, 32] 
const NBase = [4*4096, 128, 32]
const Ds = 1:3
const blockSizeBase = [
  [16, 64, 256, 1024, 4096, 2*4096, 4*4096, 8*4096],
  [2, 4, 8, 16, 32, 64, 128, 256],
  [2, 4, 6, 8, 16, 32, 64],
]

function nfft_block_size_comparison(Ds=1:3, m = 4, σ = 2.0)
  println("\n\n ##### nfft_block_size_comparison ##### \n\n")

  df = DataFrame(Package=String[], Threads=Int[], D=Int[], M=Int[], N=Int[], m = Int[], σ=Float64[],
                   TimeTrafo=Float64[], TimeAdjoint=Float64[], blocking=Bool[], blockSizeBase=Int[] )  

  for D in Ds
    @info "### Dimension D=$D ###"
    N = ntuple(d->NBase[D], D)
    M = prod(N)
    
    x = rand(D,M) .- 0.5
    fHat = randn(ComplexF64, M)
    fApprox = randn(ComplexF64, N)
    gHatApprox = randn(ComplexF64, M)

    for bl in blockSizeBase[D]
        @info "b=$(bl) D=$D"

          blockSize = ntuple(d-> bl, D)
          p = NFFTPlan(x, N; m, σ, precompute=POLYNOMIAL, blocking=true, blockSize=blockSize)

          @info "Adjoint benchmark:"
          BenchmarkTools.DEFAULT_PARAMETERS.seconds = benchmarkTime[1] 
          b = @benchmark mul!($fApprox, $(adjoint(p)), $fHat)
          tadjoint = minimum(b).time / 1e9

          @info "Trafo benchmark:"
          BenchmarkTools.DEFAULT_PARAMETERS.seconds = benchmarkTime[2]
          b = @benchmark mul!($gHatApprox, $p, $fApprox)
          ttrafo = minimum(b).time / 1e9

          push!(df, ("NFFT.jl", Threads.nthreads(), D, M, N[D], m, σ, ttrafo, tadjoint, true, bl))
    end

    # non blocking
    p = NFFTPlan(x, N; m, σ, precompute=POLYNOMIAL, blocking=false)

    @info "Adjoint benchmark:"
    BenchmarkTools.DEFAULT_PARAMETERS.seconds = benchmarkTime[1] 
    b = @benchmark mul!($fApprox, $(adjoint(p)), $fHat)
    tadjoint = minimum(b).time / 1e9

    @info "Trafo benchmark:"
    BenchmarkTools.DEFAULT_PARAMETERS.seconds = benchmarkTime[2]
    b = @benchmark mul!($gHatApprox, $p, $fApprox)
    ttrafo = minimum(b).time / 1e9

    push!(df, ("NFFT.jl", Threads.nthreads(), D, M, N[D], m, σ, ttrafo, tadjoint, false, 0))
  end
  return df
end



function plot_performance_block_size(df, Ds)

  Plots.scalefontsizes()
  Plots.scalefontsizes(1.5)
  
  colors = [:black, :orange, :blue, :green, :brown, :gray, :blue, :purple, :yellow ]
  ls = [:solid, :solid, :solid, :solid, :solid, :dash, :solid, :dash, :solid]
  shape = [:circle, :circle, :circle, :xcross, :circle, :xcross, :xcross, :circle]

  pl = Matrix{Any}(undef, 2, length(Ds))
  for (i,D) in enumerate(Ds)
    titleTrafo = L"\textrm{NFFT}, \textrm{%$(D)D}"
    titleAdjoint = L"\textrm{NFFT}^H, \textrm{%$(D)D}"

    df1_ = df[df.σ.==2.0 .&& df.D.==D .&& df.blocking .== true ,:]
    df2_ = df[df.σ.==2.0 .&& df.D.==D .&& df.blocking .== false ,:]
    maxTimeTrafo = max(maximum(df1_[:,:TimeTrafo]), maximum(df2_[:,:TimeTrafo]))*1.1
    maxTimeAdjoint = max(maximum(df1_[:,:TimeAdjoint]),maximum(df2_[:,:TimeAdjoint]))*1.1

    blockSizes = df1_[df1_.Threads .== threads[1], :blockSizeBase]

    p1 = plot(blockSizes, 
              df1_[df1_.Threads .== threads[1], :TimeTrafo], ylims=(0.0,maxTimeTrafo),
              label="1 thread", lw=2, ylabel="Runtime / s", xlabel = i==length(Ds) ? "Block Size" : "",
              legend = nothing, title=titleTrafo, shape=:circle, c=:black)

    for p=2:length(threads)      
      plot!(p1, blockSizes, 
            df1_[df1_.Threads .== threads[p],:TimeTrafo], 
              label="$(threads[p]) threads", lw=2, shape=shape[p], ls=ls[p], 
              c=colors[p], msc=colors[p], mc=colors[p], ms=4, msw=2)
    end

    plot!(p1, blockSizes, 
              df2_[df2_.Threads .== threads[1], :TimeTrafo].*ones(length(blockSizes)), ylims=(0.0,maxTimeTrafo),
              label="regular: 1 thread", lw=2,  ls=:dash,
              c=colors[1])

    for p=2:length(threads)      
      plot!(p1, blockSizes, 
              df2_[df2_.Threads .== threads[p], :TimeTrafo].*ones(length(blockSizes)), 
              label="regular: $(threads[p]) threads", lw=2,  ls=:dash, 
              c=colors[p])
    end

    p2 = plot(blockSizes, 
              df1_[df1_.Threads .== threads[1],:TimeAdjoint], ylims=(0.0,maxTimeAdjoint),
              lw=2, xlabel = i==length(Ds) ? "Block Size" : "", label="1 thread",
              legend = i==2 ? :topright : nothing, title=titleAdjoint, shape=:circle, c=:black)

    for p=2:length(threads)      
      plot!(p2, blockSizes, 
            df1_[df1_.Threads .== threads[p],:TimeAdjoint], 
              label="$(threads[p]) threads", lw=2, shape=shape[p], ls=ls[p], 
              c=colors[p], msc=colors[p], mc=colors[p], ms=4, msw=2)
    end

    plot!(p2, blockSizes, 
              df2_[df2_.Threads .== threads[1], :TimeAdjoint].*ones(length(blockSizes)), ylims=(0.0,maxTimeAdjoint),
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
  if false
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









