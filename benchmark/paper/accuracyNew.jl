using NFFT, DataFrames, LinearAlgebra, Statistics, LaTeXStrings, DelimitedFiles 
using Plots; pgfplotsx()

include("../../Wrappers/NFFT3.jl")
include("../../Wrappers/FINUFFT.jl")


const packagesCtor = [NFFTPlan, NFFTPlan, NFFTPlan, NFFTPlan,  NFFT3Plan, NFFT3Plan, FINUFFTPlan]
const packagesStr = ["NFFT.jl/FULL", "NFFT.jl/LINEAR", "NFFT.jl/TENSOR", "NFFT.jl/POLY", "NFFT3/LINEAR", "NFFT3/TENSOR", "FINUFFT"]
const precomp = [NFFT.FULL, NFFT.LINEAR, NFFT.TENSOR, NFFT.POLYNOMIAL, NFFT.LINEAR, NFFT.TENSOR, NFFT.LINEAR]
const blocking = [false, true, true, true, false, false, false]

const σs = range(1.25, 4, length=12)
const ms = 3:10
const NBase = [512*512, 512, 64]
const Ds = 1:3
const P = 10

function nfft_accuracy_comparison(Ds, σs, ms)
  println("\n\n ##### nfft_accuracy_comparison ##### \n\n")

  df = DataFrame(Package=String[], D=Int[], J=Int[], N=Int[], m = Int[], σ=Float64[],
                   ErrorL2Trafo=Float64[], ErrorL2Adjoint=Float64[], ErrorInfTrafo=Float64[], ErrorInfAdjoint=Float64[] )  

  Random.seed!(1) 

  for p = 1:P
    @info p
    for D in Ds
        N = ntuple(d->NBase[D], D)
        J = prod(N)
        
        for σ in σs
          for m in ms
            @info "m=$m D=$D σ=$σ "
            k = rand(D,J) .- 0.5
            fHat = randn(ComplexF64, J)
            f = rand(ComplexF64, N)

            # ground truth (numerical)
            #pNDFT = NDFTPlan(k, N)
            pNDFT = NFFT3Plan(k, N; m=12, σ=2)
            g = adjoint(pNDFT) * fHat
            gHat = pNDFT * f

            for pl = 1:length(packagesStr)
              planner = packagesCtor[pl]
              if planner != FINUFFT || σ == 2.0 # FINUFFT is not included in sigma sweep
                p = planner(k, N; m, σ, precompute=precomp[pl], blocking=blocking[pl])

                gApprox = adjoint(p) * fHat
                gHatApprox = p * f

                eL2adjoint = norm(g[:] - gApprox[:], 2) / norm(g[:], 2)
                eL2trafo = norm(gHat[:] - gHatApprox[:], 2) / norm(gHat[:], 2)
                eInfadjoint = norm(g[:] - gApprox[:], Inf) / norm(g[:], Inf)
                eInftrafo = norm(gHat[:] - gHatApprox[:], Inf) / norm(gHat[:], Inf)
                
                push!(df, (packagesStr[pl], D, J, N[D], m, σ, eL2trafo, eL2adjoint, eInftrafo, eInfadjoint))
              end
          end
        end
      end
    end
  end
  return df
end

function plot_accuracy_m(df, packagesStr, packagesStrShort, filename, D=1,
      colors = [RGB(0.0,0.29,0.57), RGB(0.3,0.5,0.7), RGB(0.94,0.53,0.12), RGB(0.99,0.75,0.05)])

  σs = range(1.25, 4, length=12)

  df1 = df[df.σ.==2.0 .&& df.D.==D,:]
  #df2_ = df[df.m.==8 .&& df.D.==D,:]

  Plots.scalefontsizes()
  Plots.scalefontsizes(1.5)
  
  gdf = groupby(df1, Cols(:m, :Package))
  df1_ = combine(gdf, :ErrorL2Trafo => mean => :ErrorL2Trafo, 
                      :ErrorL2Adjoint => mean => :ErrorL2Adjoint,
                      :ErrorInfTrafo => mean => :ErrorInfTrafo,
                      :ErrorInfAdjoint => mean => :ErrorInfAdjoint)

  ls = [:solid, :solid, :solid, :solid]
  shape = [:circle, :xcross, :cross, :xcross ]

  p1 = plot(ms, df1_[df1_.Package.==packagesStr[1],:ErrorL2Trafo], 
            yscale = :log10, label=packagesStrShort[1], lw=2, xlabel = L"m", ylabel="Relative Error L2",
            legend = (:topright), title=L"\textrm{NFFT}", 
            shape=shape[1], ls=ls[1], ylims = (1e-15,2e-5), xlims = (3,10),
            c=colors[1], msc=colors[1], mc=colors[1],ms=5, msw=2)

  for p=2:length(packagesStr)      
    plot!(p1, ms, df1_[df1_.Package.==packagesStr[p],:ErrorL2Trafo], 
            yscale = :log10, label=packagesStrShort[p], lw=2, shape=shape[p], ls=ls[p], 
            c=colors[p], msc=colors[p], mc=colors[p], ms=5, msw=2)
  end

  p1_ = plot(ms, df1_[df1_.Package.==packagesStr[1],:ErrorInfTrafo], 
            yscale = :log10, label=packagesStrShort[1], lw=2, xlabel = L"m", ylabel="Relative Error Inf",
            legend = nothing, #(:topright), #title=L"\textrm{NFFT}", 
            shape=shape[1], ls=ls[1], ylims = (1e-15,2e-5), xlims = (3,10),
            c=colors[1], msc=colors[1], mc=colors[1],ms=5, msw=2)

  for p=2:length(packagesStr)      
    plot!(p1_, ms, df1_[df1_.Package.==packagesStr[p],:ErrorInfTrafo], 
            yscale = :log10, label=nothing, lw=2, shape=shape[p], ls=:solid, 
            c=colors[p], msc=colors[p], mc=colors[p], ms=5, msw=2)
  end

  p2 = plot(ms, df1_[df1_.Package.==packagesStr[1],:ErrorL2Adjoint], 
            yscale = :log10, lw=2, xlabel = L"m", #ylabel="Relative Error",
            legend = nothing, title=L"\textrm{NFFT}^H", 
            shape=shape[1], ls=ls[1], ylims = (1e-15,2e-5), xlims = (3,10),
            c=colors[1], msc=colors[1], mc=colors[1], ms=5, msw=2)

  for p=2:length(packagesStr)      
    plot!(p2, ms, df1_[df1_.Package.==packagesStr[p],:ErrorL2Adjoint], 
            yscale = :log10, lw=2, shape=shape[p], ls=ls[p], 
            c=colors[p], msc=colors[p], mc=colors[p], ms=5, msw=2)
  end

  p2_ = plot(ms, df1_[df1_.Package.==packagesStr[1],:ErrorInfAdjoint], 
            yscale = :log10, lw=2, xlabel = L"m", #ylabel="Relative Error",
            legend = nothing, #title=L"\textrm{NFFT}^H", 
            shape=shape[1], ls=ls[1], ylims = (1e-15,2e-5), xlims = (3,10),
            c=colors[1], msc=colors[1], mc=colors[1], ms=5, msw=2)

  for p=2:length(packagesStr)      
    plot!(p2_, ms, df1_[df1_.Package.==packagesStr[p],:ErrorInfAdjoint], 
            yscale = :log10, lw=2, shape=shape[p], ls=:solid, 
            c=colors[p], msc=colors[p], mc=colors[p], ms=5, msw=2)
  end

  p = plot(p1, p2, p1_, p2_, layout=(2,2), size=(800,450), dpi=200, tex_output_standalone = true)
  #p = plot(p1, layout=(1,2), size=(800,450), dpi=200)

  mkpath("./img/")
  savefig(p, joinpath("./img/",filename*".pdf"))
  savefig(p, joinpath("./img/",filename*".tex"))
  return p
end



dfm = nfft_accuracy_comparison(2, [2.0], ms)
writedlm("data/accuracy_m_.csv", Iterators.flatten(([names(dfm)], eachrow(dfm))), ',')

data, header = readdlm("data/accuracy_m_.csv", ',', header=true);
dfm = DataFrame(data, vec(header))

plot_accuracy_m(dfm, ["NFFT.jl/TENSOR", "NFFT3/TENSOR", "FINUFFT"],
                     ["NFFT.jl", "NFFT3", "FINUFFT"], "accuracy_m_D2_", 2,
                     [RGB(0.0,0.29,0.57), RGB(0.94,0.53,0.12), RGB(0.99,0.75,0.05)])
plot_accuracy_m(dfm, ["NFFT.jl/FULL", "NFFT.jl/TENSOR", "NFFT.jl/LINEAR", "NFFT.jl/POLY"], 
                     ["FULL", "TENSOR", "LINEAR", "POLYNOMIAL"],
                      "accuracy_m_pre_D2_", 2,
                      [RGB(0.7,0.13,0.16), RGB(0.3,0.5,0.7), RGB(0.5,0.48,0.45) ,RGB(0.0,0.29,0.57)])

@info "Mean error deviation  NFFT.jl / NFFT3"
mean((dfm[dfm.Package.=="NFFT.jl/POLY",:ErrorL2Trafo] ./ dfm[dfm.Package.=="NFFT3/TENSOR",:ErrorL2Trafo])[1:5])

@info "Mean error deviation  FINUFFT / NFFT.jl"
mean((dfm[dfm.Package.=="FINUFFT",:ErrorL2Trafo] ./ dfm[dfm.Package.=="NFFT.jl/POLY",:ErrorL2Trafo])[1:5])








