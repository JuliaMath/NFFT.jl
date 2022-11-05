using LinearAlgebra
import ducc0_jll

const libducc = ducc0_jll.libducc_julia

 function ducc_nu2u(coord::Array{Cdouble,2}, data::Vector{Complex{Cdouble}}, res::Array{Complex{Cdouble},D}; epsilon::AbstractFloat, nthreads::Int=1, 
                    verbosity::Int=0, periodicity::AbstractFloat=1., forward::Int=0) where D
   shape = size(res)
   shp = Array{Csize_t}([x for x in shape])
   ccall((:nufft_nu2u_julia_double,libducc),
     Cvoid, (Csize_t,Csize_t,Ptr{Csize_t},Ptr{Cdouble},Ptr{Cdouble},Cint,Cdouble,Csize_t,Ptr{Cdouble},Csize_t,Cdouble,Cdouble,Cdouble,Cint),
     size(coord)[1], size(coord)[2], pointer(shp), pointer(data), pointer(coord),
     forward, epsilon, nthreads, pointer(res), verbosity, 1.1, 2.6,
     periodicity, 0)
   return res
 end

 function ducc_u2nu(coord::Array{Cdouble,2}, data::Array{Complex{Cdouble},D}, res::Vector{Complex{Cdouble}}; epsilon::AbstractFloat, nthreads::Int=1, 
                    verbosity::Int=0, periodicity::AbstractFloat=1., forward::Int=1) where D
   shape = size(data)
   shp = Array{Csize_t}([x for x in shape])
   ccall((:nufft_u2nu_julia_double,libducc),
     Cvoid, (Csize_t,Csize_t,Ptr{Csize_t},Ptr{Cdouble},Ptr{Cdouble},Cint,Cdouble,Csize_t,Ptr{Cdouble},Csize_t,Cdouble,Cdouble,Cdouble,Cint),
     size(coord)[1], size(coord)[2], pointer(shp), pointer(data), pointer(coord),
     forward, epsilon, nthreads, pointer(res), verbosity, 1.1, 2.6,
     periodicity, 0)
   return res
 end


mutable struct DUCC0Plan{T,D} <: AbstractNFFTPlan{T,D,1} 
  N::NTuple{D,Int64}
  J::Int64
  k::Matrix{T}
  m::Int
  σ::T
  reltol::T
end

################
# constructors
################

function DUCC0Plan(k::Matrix{T}, N::NTuple{D,Int}; 
              dims::Union{Integer,UnitRange{Int64}}=1:D,
              kargs...) where {D,T}

  if dims != 1:D
    error("DUCC0Plan directional plans not yet implemented!")
  end

  J = size(k,2)

  m, σ, reltol = accuracyParams(; kargs...)

  k_ = k

  reltol = max(reltol, 1.0e-15)

  p = DUCC0Plan(N, J, k_, m, T(σ), T(reltol))

  return p
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

function LinearAlgebra.mul!(fHat::StridedArray, p::DUCC0Plan{T,D}, f::AbstractArray;
             verbose=false, timing::Union{Nothing,TimingStats} = nothing) where {T,D}

  ducc_u2nu(p.k, f, fHat;
     epsilon = p.reltol, nthreads=1, verbosity=Int64(verbose), periodicity=1.0, forward=1)

  return fHat
end

function LinearAlgebra.mul!(f::StridedArray, pl::Adjoint{Complex{T},<:DUCC0Plan{T,D}}, fHat::AbstractArray;
                     verbose=false, timing::Union{Nothing,TimingStats} = nothing) where {T,D}
  p = pl.parent

  ducc_nu2u(p.k, fHat, f;
     epsilon = p.reltol, nthreads=1, verbosity=Int64(verbose), periodicity=1.0, forward=0)

  return f
end
