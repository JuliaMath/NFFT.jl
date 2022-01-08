using NFFT, DataFrames, Plots, LinearAlgebra, FFTW, BenchmarkTools
import NFFT3

include("NFFT3.jl")

function nfft_accuracy_comparison()
  println("\n\n ##### nfft_accuracy_comparison ##### \n\n")

  df = DataFrame(Package=String[], D=Int[], M=Int[], N=Int[], m = Int[], σ=Float64[],
                   ErrorTrafo=Float64[], ErrorAdjoint=Float64[] )  
  N = [256, 64]

  for D = 1:2
  
      NN = ntuple(d->N[D], D)
      M = prod(NN)
      
      for σ in [1.25, 1.5, 2.0]
        for m = 1:14
          @info "D=$D  σ=$σ  m=$m "
          x = rand(D,M) .- 0.5
          fHat = randn(ComplexF64, M)

          p = plan_nfft(x, NN; m, σ, precompute=NFFT.FULL)
          pNDFT = NDFTPlan(x, NN)
          f = ndft_adjoint(pNDFT, fHat)
          fApprox = nfft_adjoint(p, fHat)
          eadjoint = norm(f[:] - fApprox[:]) / norm(f[:])

          gHat = ndft(pNDFT, f)
          gHatApprox = nfft(p, f)
          etrafo = norm(gHat[:] - gHatApprox[:]) / norm(gHat[:])
          
          push!(df, ("NFFT.jl", D, M, N[D], m, σ, etrafo, eadjoint))

          p = NFFT3Plan(reshape(x,D,:), NN; m, σ, precompute=NFFT.FULL)
          fApprox = nfft_adjoint(p, fHat)
          eadjoint = norm(f[:] - fApprox[:]) / norm(f[:])

          gHatApprox = nfft(p, f)
          etrafo = norm(gHat[:] - gHatApprox[:]) / norm(gHat[:])
          
          push!(df, ("NFFT3", D, M, N[D], m, σ, etrafo, eadjoint))
      end
    end
  end
  return df
end



function plot_accuracy(df, D=1)

  σs = [1.25, 1.5, 2.0]

  plots = Matrix{Any}(undef, length(σs), 2)

  m = 1:14

  for (i,σ) in enumerate(σs)
    df_ = df[df.σ.==σ .&& df.D.==D,:]

    p1 = plot(m, df_[df_.Package.=="NFFT.jl",:ErrorTrafo], 
              yscale = :log10, label="NFFT.jl", lw=2, xlabel = "m", title="Trafo σ=$(σ)")
    plot!(p1, m, df_[df_.Package.=="NFFT3",:ErrorTrafo], 
          yscale = :log10, label="NFFT3", lw=2)

    plots[i,1] = p1

    p2 = plot(m, df_[df_.Package.=="NFFT.jl",:ErrorAdjoint], 
          yscale = :log10, label="NFFT.jl", lw=2, xlabel = "m", title="Adjoint σ=$(σ)")
    plot!(p2, m, df_[df_.Package.=="NFFT3",:ErrorAdjoint], 
      yscale = :log10, label="NFFT3", lw=2)

    plots[i,2] = p2
  end

  p = plot(plots..., layout=(2,length(σs)))
  savefig(p, "accuracy_D$(D).png")
end

#df = nfft_accuracy_comparison()
#plot_accuracy(df, 1)
#plot_accuracy(df, 2)




const LUTSize = 20000

FFTW.set_num_threads(Threads.nthreads())
ccall(("omp_set_num_threads",NFFT3.lib_path_nfft),Nothing,(Int64,),convert(Int64,Threads.nthreads()))
@info ccall(("nfft_get_num_threads",NFFT3.lib_path_nfft),Int64,())
if Threads.nthreads() == 1
  NFFT._use_threads[] = false
end

function nfft_performance_comparison(m = 5, σ = 2.0)
  println("\n\n ##### nfft_performance_comparison ##### \n\n")

  df = DataFrame(Package=String[], D=Int[], M=Int[], N=Int[], 
                   Undersampled=Bool[], Pre=String[], m = Int[], σ=Float64[],
                   TimePre=Float64[], TimeTrafo=Float64[], TimeAdjoint=Float64[] )  

  preString = ["LUT", "FULL", "FULL_LUT"]
  preNFFTjl = [NFFT.LUT, NFFT.FULL, NFFT.FULL_LUT]
  N = [collect(4096* (4 .^(0:3))),collect(128* (2 .^ (0:3))),[32,48,64,72]]

  for D = 2:2
    for U = 4:4
      NN = ntuple(d->N[D][U], D)
      M = prod(NN) #*2 #÷ 10

      for pre = 1:2

        @info D, NN, M, pre
        
        T = Float64

        x = T.(rand(T,D,M) .- 0.5)
        fHat = randn(Complex{T}, M)

        tpre = @elapsed p = plan_nfft(x, NN; m, σ, window=:kaiser_bessel, LUTSize, precompute=preNFFTjl[pre], sortNodes=false, fftflags=FFTW.ESTIMATE)
        f = similar(fHat, p.N)
        tadjoint = @elapsed fApprox = nfft_adjoint!(p, fHat, f)
        ttrafo = @elapsed nfft!(p, fApprox, fHat)
        
        push!(df, ("NFFT.jl", D, M, N[D][U], false, preString[pre], m, σ,
                   tpre, ttrafo, tadjoint))

        tpre = @elapsed p = NFFT3Plan(x, NN; m, σ, window=:kaiser_bessel, LUTSize, precompute=preNFFTjl[pre], sortNodes=true, fftflags=FFTW.ESTIMATE)
        tadjoint = @elapsed fApprox = nfft_adjoint!(p, fHat, f)
        ttrafo = @elapsed nfft!(p, fApprox, fHat)
                
        push!(df, ("NFFT3", D, M, N[D][U], false, preString[pre], m, σ,
                    tpre, ttrafo, tadjoint))
          
      end
    end
  end
  return df
end

df = nfft_performance_comparison(3, 2.0)



# writedlm("test.csv", Iterators.flatten(([names(iris)], eachrow(iris))), ',')
#
#  julia> using DelimitedFiles, DataFrames
#
# julia> data, header = readdlm(joinpath(dirname(pathof(DataFrames)),
# "..", "docs", "src", "assets", "iris.csv"),
# ',', header=true);
#
#julia> iris_raw = DataFrame(data, vec(header))
#
#