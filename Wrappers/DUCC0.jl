using LinearAlgebra
using Ducc0


mutable struct Ducc0NufftPlan{T,D} <: AbstractNFFTPlan{T,D,1}
  N::NTuple{D,Int64}
  J::Int64
  plan::Ducc0.Nufft.NufftPlan
end

function Ducc0NufftPlan(
  k::Matrix{T},
  N::NTuple{D,Int};
  kargs...
) where {D,T<:Float64}
  J = size(k, 2)
  sigma_min = 1.1
  sigma_max = 2.6

  m, σ, reltol = accuracyParams(; kargs...)

  reltol = max(reltol, 1e-15 * (10)^D ) # otherwise ducc0 will crash
  nthreads = Threads.nthreads()
  sigma_min = σ-0.026 # try match σ   # 1.1
  sigma_max =  σ+0.026 # try match σ  # 2.6

  reltol = max(
      reltol,
      1.1 * Ducc0.Nufft.best_epsilon(D, false, sigma_min = sigma_min, sigma_max = sigma_max),
  )

  plan = Ducc0.Nufft.make_plan(
      k,
      N,
      epsilon = reltol,
      nthreads = nthreads,
      sigma_min = sigma_min,
      sigma_max = sigma_max,
      periodicity = 1.0,
      fft_order = false,
  )
  return Ducc0NufftPlan{T,D}(N, J, plan)
end

function Base.show(io::IO, p::Ducc0NufftPlan)
  print(io, "Ducc0NufftPlan")
end

AbstractNFFTs.size_in(p::Ducc0NufftPlan) = Int.(p.N)
AbstractNFFTs.size_out(p::Ducc0NufftPlan) = (Int(p.J),)

function AbstractNFFTs.plan_nfft(
  ::Type{<:Array},
  k::Matrix{T},
  N::NTuple{D,Int},
  rest...;
  kargs...,
) where {T,D}
  return Ducc0NufftPlan(k, N, rest...; kargs...)
end

function LinearAlgebra.mul!(
  fHat::Vector{Complex{T}},
  p::Ducc0NufftPlan{T,D},
  f::Array{Complex{T},D};
  verbose = false,
) where {T<:Float64, D}
  Ducc0.Nufft.u2nu_planned!(p.plan, f, fHat, forward = true, verbose = verbose)
  return fHat
end

function LinearAlgebra.mul!(
  f::Array{Complex{T},D},
  pl::AbstractNFFTs.Adjoint{Complex{T},<:Ducc0NufftPlan{T,D}},
  fHat::Vector{Complex{T}};
  verbose = false,
) where {T<:Float64, D}
  p = pl.parent

  Ducc0.Nufft.nu2u_planned!(p.plan, fHat, f, forward = false, verbose = verbose)
  return f
end