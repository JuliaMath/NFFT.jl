using LinearAlgebra
import ducc0_jll

const libducc = ducc0_jll.libducc_julia

mutable struct DUCC0Plan{T,D} <: AbstractNFFTPlan{T,D,1} 
  N::NTuple{D,Int64}
  J::Int64
#  k::Matrix{T}      # stored by the library itself
#  m::Int
#  σ::T
  reltol::T
  cplan::Ptr{Cvoid}
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

  reltol = max(reltol, 1.0e-14)
  nthreads = 1
  sigma_min = 1.1
  sigma_max = 2.6
  periodicity = 1.
  fft_order = 0
  shape = Array{Csize_t}([x for x in N])

  ptr = ccall((:make_nufft_plan_double,libducc), Ptr{Cvoid}, (Cint, Csize_t, Csize_t, Ptr{Csize_t}, Ptr{Cdouble}, Cdouble, Csize_t, Cdouble, Cdouble, Cdouble, Cint), 0, D, J, pointer(shape), pointer(k), reltol, nthreads, sigma_min, sigma_max, periodicity, fft_order)

#  p = DUCC0Plan(N, J, m, T(σ), T(reltol), ptr)
  p = DUCC0Plan(N, J, T(reltol), ptr)

  finalizer(p -> begin
    println("finalize!")
    ccall((:delete_nufft_plan_double, libducc), Cvoid, (Ptr{Cvoid},), p.cplan)
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
             verbose=true, timing::Union{Nothing,TimingStats} = nothing) where {T,D}

  nthreads = 1
  forward = 1
  ccall((:planned_u2nu,libducc),
    Cvoid, (Ptr{Cvoid}, Cint, Csize_t, Ptr{Cdouble}, Ptr{Cdouble}),
    p.cplan, forward, Int64(verbose), pointer(f), pointer(fHat))
#  ccall((:nufft_u2nu_julia_double,libducc),
#     Cvoid, (Csize_t,Csize_t, Ref{NTuple{D,Csize_t}},Ptr{Cdouble},Ptr{Cdouble},Cint,Cdouble,Csize_t,Ptr{Cdouble},Csize_t,Cdouble,Cdouble,Cdouble,Cint),
#     D, p.J, size(f), pointer(f), pointer(p.k),
#     forward, p.reltol, nthreads, pointer(fHat), Int64(verbose), 1.99, 2.001,
#     1.0, 0)

  return fHat
end

function LinearAlgebra.mul!(f::StridedArray, pl::Adjoint{Complex{T},<:DUCC0Plan{T,D}}, fHat::AbstractArray;
                     verbose=true, timing::Union{Nothing,TimingStats} = nothing) where {T,D}
  p = pl.parent

  nthreads = 1
  forward = 0
  ccall((:planned_nu2u,libducc),
    Cvoid, (Ptr{Cvoid}, Cint, Csize_t, Ptr{Cdouble}, Ptr{Cdouble}),
    p.cplan, forward, Int64(verbose), pointer(fHat), pointer(f))
#  ccall((:nufft_nu2u_julia_double,libducc),
#     Cvoid, (Csize_t,Csize_t, Ref{NTuple{D,Csize_t}},Ptr{Cdouble},Ptr{Cdouble},Cint,Cdouble,Csize_t,Ptr{Cdouble},Csize_t,Cdouble,Cdouble,Cdouble,Cint),
#     D, p.J, size(f), pointer(fHat), pointer(p.k),
#     forward, p.reltol, nthreads, pointer(f), Int64(verbose), 1.99, 2.001,
#     1.0, 0)

  return f
end
