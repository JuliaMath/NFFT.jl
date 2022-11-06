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
  cplan::Ptr{Cvoid}
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

  reltol = max(reltol, 1e-15 * (10)^D ) # otherwise ducc0 will crash
  nthreads = Threads.nthreads()
  sigma_min = σ-0.026 # try match σ   # 1.1
  sigma_max =  σ+0.026 # try match σ  # 2.6
  periodicity = 1.0
  fft_order = 0

  ptr = ccall((:make_nufft_plan_double,libducc), Ptr{Cvoid}, 
                (Cint, Csize_t, Csize_t, Ref{NTuple{D,Csize_t}}, Ptr{Cdouble}, Cdouble, Csize_t, Cdouble, Cdouble, Cdouble, Cint), 
                0, D, J, N, pointer(k), reltol, nthreads, sigma_min, sigma_max, periodicity, fft_order)

  p = DUCC0Plan(N, J, k, m, T(σ), T(reltol), ptr)

  finalizer(p -> begin
    # println("finalize!")
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

function LinearAlgebra.mul!(fHat::Vector{Complex{T}}, p::DUCC0Plan{T,D}, f::Array{Complex{T},D};
             verbose=false, timing::Union{Nothing,TimingStats} = nothing) where {T<:Float64,D}

  forward = 1
  ccall((:planned_u2nu,libducc),
         Cvoid, (Ptr{Cvoid}, Cint, Csize_t, Ptr{Cdouble}, Ptr{Cdouble}),
         p.cplan, forward, Int64(verbose), pointer(f), pointer(fHat))

  return fHat
end

function LinearAlgebra.mul!(f::Array{Complex{T},D}, pl::Adjoint{Complex{T},<:DUCC0Plan{T,D}}, fHat::Vector{Complex{T}};
                     verbose=false, timing::Union{Nothing,TimingStats} = nothing) where {T<:Float64,D}
  p = pl.parent

  forward = 0
  ccall((:planned_nu2u,libducc),
         Cvoid, (Ptr{Cvoid}, Cint, Csize_t, Ptr{Cdouble}, Ptr{Cdouble}),
         p.cplan, forward, Int64(verbose), pointer(fHat), pointer(f))

  return f
end
