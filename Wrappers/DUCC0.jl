using LinearAlgebra
import ducc0_jll

const libducc = ducc0_jll.libducc_julia

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
  reltol

  reltol = max(reltol, 1e-15 * (10)^D ) # otherwise ducc0 will crash

  p = DUCC0Plan(N, J, k, m, T(σ), T(reltol))

  finalizer(p -> begin
    #println("Run DUCC0Plan finalizer")
  end, p)

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

  nthreads = Threads.nthreads()
  forward = 1
  ccall((:nufft_u2nu_julia_double,libducc),
     Cvoid, (Csize_t,Csize_t, Ref{NTuple{D,Csize_t}},Ptr{Cdouble},Ptr{Cdouble},Cint,Cdouble,Csize_t,Ptr{Cdouble},Csize_t,Cdouble,Cdouble,Cdouble,Cint),
     D, p.J, size(f), pointer(f), pointer(p.k),
     forward, p.reltol, nthreads, pointer(fHat), Int64(verbose), p.σ-0.026, p.σ+0.026, #1.99, 2.001,
     1.0, 0)

  return fHat
end

function LinearAlgebra.mul!(f::StridedArray, pl::Adjoint{Complex{T},<:DUCC0Plan{T,D}}, fHat::AbstractArray;
                     verbose=false, timing::Union{Nothing,TimingStats} = nothing) where {T,D}
  p = pl.parent

  nthreads = Threads.nthreads()
  forward = 0
  ccall((:nufft_nu2u_julia_double,libducc),
     Cvoid, (Csize_t,Csize_t, Ref{NTuple{D,Csize_t}},Ptr{Cdouble},Ptr{Cdouble},Cint,Cdouble,Csize_t,Ptr{Cdouble},Csize_t,Cdouble,Cdouble,Cdouble,Cint),
     D, p.J, size(f), pointer(fHat), pointer(p.k),
     forward, p.reltol, nthreads, pointer(f), Int64(verbose), p.σ-0.026, p.σ+0.026, #1.99, 2.001,
     1.0, 0)

  return f
end
