using Ducc0
using LinearAlgebra

mutable struct DUCC0Plan{T,D} <: AbstractNFFTPlan{T,D,1}
  N::NTuple{D,Int64}
  J::Int64
  k::Matrix{T}
  m::Int
  σ::T
  reltol::T
  plan::Ducc0.Nufft.NufftPlan
end

################
# constructors
################

function DUCC0Plan(k::Matrix{T}, N::NTuple{D,Int}; 
              dims::Union{Integer,UnitRange{Int64}}=1:D,
              kargs...) where {D,T<:Float64}

  if dims != 1:D
    error("DUCC0Plan directional plans not yet implemented!")
  end

  J = size(k,2)

  m, σ, reltol = accuracyParams(; kargs...)

  # simple switch whether or not to follow the requested oversampling factor
  # Free choice of oversampling factor improves performance in many cases
  # and allows better accuracy in others.
  match_sigma = true
  if match_sigma
    sigma_min = σ - 0.026 # try match σ
    sigma_max = σ + 0.026 # try match σ
  else
    sigma_min = 1.1
    sigma_max = 2.6
  end
  reltol = max(reltol, 1.1*Ducc0.Nufft.best_epsilon(UInt64(D), false, sigma_min=sigma_min, sigma_max=sigma_max))

  plan = Ducc0.Nufft.make_plan(
    k,
    N,
    epsilon=reltol,
    nthreads=UInt64(Threads.nthreads()),
    sigma_min=sigma_min,
    sigma_max=sigma_max,
    periodicity=1.0,
    fft_order=false,
  )
  return DUCC0Plan(N, J, k, m, T(σ), T(reltol), plan)
end


function Base.show(io::IO, p::DUCC0Plan)
  print(io, "DUCC0Plan")
end

AbstractNFFTs.size_in(p::DUCC0Plan) = Int.(p.N)
AbstractNFFTs.size_out(p::DUCC0Plan) = (Int(p.J),)

function AbstractNFFTs.plan_nfft(::Type{<:Array}, k::Matrix{T}, N::NTuple{D,Int}, rest...;
                   timing::Union{Nothing,TimingStats} = nothing, kargs...) where {T,D}
  t = @elapsed begin
    p = DUCC0Plan(k, N, rest...; kargs...)
  end
  if timing != nothing
    timing.pre = t
  end
  return p
end

function LinearAlgebra.mul!(fHat::Vector{Complex{T}}, p::DUCC0Plan{T,D}, f::Array{Complex{T},D};
             verbose=false, timing::Union{Nothing,TimingStats} = nothing) where {T<:Float64,D}

  Ducc0.Nufft.u2nu_planned!(p.plan, f, fHat, forward=true, verbose=verbose)
  return fHat
end

function LinearAlgebra.mul!(f::Array{Complex{T},D}, pl::Adjoint{Complex{T},<:DUCC0Plan{T,D}}, fHat::Vector{Complex{T}};
                     verbose=false, timing::Union{Nothing,TimingStats} = nothing) where {T<:Float64,D}
  p = pl.parent

  Ducc0.Nufft.nu2u_planned!(p.plan, fHat, f, forward=false, verbose=verbose)
  return f
end
