using LinearAlgebra
import FINUFFT

mutable struct FINUFFTPlan{T,D} <: AbstractNFFTPlan{T,D,1} 
  N::NTuple{D,Int64}
  J::Int64
  k::Matrix{T}
  m::Int
  σ::T
  reltol::T
  fftflags::UInt32
  planTrafo::FINUFFT.finufft_plan{T}
  planAdjoint::FINUFFT.finufft_plan{T}
end

################
# constructors
################

function FINUFFTPlan(k::Matrix{T}, N::NTuple{D,Int}; 
              dims::Union{Integer,UnitRange{Int64}}=1:D,
              fftflags=UInt32(NFFT.FFTW.ESTIMATE), 
              kargs...) where {D,T}

  if dims != 1:D
  error("FINUFFT directional plans not yet implemented!")
  end

  J = size(k,2)

  m, σ, reltol = accuracyParams(; kargs...)

  reltol = max(reltol, 1.0e-15)

  k_ = T.(2π) * k 
  nodes = ntuple(d->vec(k_[d,:]), D)

  planTrafo = FINUFFT.finufft_makeplan(2, collect(N), -1, 1, reltol;
                          nthreads = Threads.nthreads(), dtype=T,
                          fftw = fftflags, upsampfac=2.0, debug=0)

  FINUFFT.finufft_setpts!(planTrafo, nodes...)

  planAdjoint = FINUFFT.finufft_makeplan(1, collect(N), 1, 1, reltol;
                          nthreads = Threads.nthreads(), dtype=T, 
                          fftw = fftflags, upsampfac=2.0, debug=0)

  FINUFFT.finufft_setpts!(planAdjoint, nodes...)

  p = FINUFFTPlan(N, J, k_, m, T(σ), T(reltol), fftflags, planTrafo, planAdjoint)

  finalizer(p -> begin
    #println("Run FINUFFT finalizer")
    FINUFFT.finufft_destroy!(p.planTrafo)
    FINUFFT.finufft_destroy!(p.planAdjoint)
  end, p)

  return p
end


function Base.show(io::IO, p::FINUFFTPlan)
  print(io, "FINUFFTPlan")
end

AbstractNFFTs.size_in(p::FINUFFTPlan) = Int.(p.N)
AbstractNFFTs.size_out(p::FINUFFTPlan) = (Int(p.J),)

function AbstractNFFTs.plan_nfft(::Type{<:Array}, k::Matrix{T}, N::NTuple{D,Int}, rest...;
                   timing::Union{Nothing,TimingStats} = nothing, kargs...) where {T,D}
  t = @elapsed begin
    p = FINUFFTPlan(k, N, rest...; kargs...)
  end
  if timing != nothing
    timing.pre = t
  end
  return p
end

function LinearAlgebra.mul!(fHat::StridedArray, p::FINUFFTPlan{T,D}, f::AbstractArray;
             verbose=false, timing::Union{Nothing,TimingStats} = nothing) where {T,D}

  FINUFFT.finufft_exec!(p.planTrafo, f, fHat)

  return fHat
end

function LinearAlgebra.mul!(f::StridedArray, pl::Adjoint{Complex{T},<:FINUFFTPlan{T,D}}, fHat::AbstractArray;
                     verbose=false, timing::Union{Nothing,TimingStats} = nothing) where {T,D}
  p = pl.parent

  FINUFFT.finufft_exec!(p.planAdjoint, fHat, f)

  return f
end



mutable struct FINNUFFTPlan{T} <: AbstractNNFFTPlan{T,1,1} 
  N::Int64
  J::Int64
  k::Matrix{T}
  y::Matrix{T}
  m::Int
  σ::T
  reltol::T
  fftflags::UInt32
end

function FINNUFFTPlan(k::Matrix{T}, y::Matrix{T}; 
  fftflags=UInt32(NFFT.FFTW.ESTIMATE), 
  kargs...) where {T}

  N = size(y,2)
  J = size(k,2)

  m, σ, reltol = accuracyParams(; kargs...)

  reltol = max(reltol, 1.0e-15)

  k_ = 2π * k 
  y_ = y 

  p = FINNUFFTPlan(N, J, k_, y_, m, T(σ), reltol, fftflags)

  return p
end


function Base.show(io::IO, p::FINNUFFTPlan)
  print(io, "FINNUFFTPlan")
end

AbstractNFFTs.size_in(p::FINNUFFTPlan) = (Int.(p.N),)
AbstractNFFTs.size_out(p::FINNUFFTPlan) = (Int(p.J),)

function LinearAlgebra.mul!(fHat::StridedArray, p::FINNUFFTPlan{T}, f::AbstractArray;
 verbose=false, timing::Union{Nothing,TimingStats} = nothing) where {T}

  D = size(p.k,1)

  forwardPlan = FINUFFT.finufft_makeplan(3,D,-1,1,p.reltol;
                              nthreads = Threads.nthreads(), 
                              fftw = p.fftflags)

  nodesX = ntuple(d->vec(p.k[d,:]), D)
  nodesY = ntuple(d->vec(p.y[d,:]), D)

  if D==1
    FINUFFT.finufft_setpts!(forwardPlan, nodesY..., T[], T[], nodesX..., T[], T[])
  elseif D==2
    FINUFFT.finufft_setpts!(forwardPlan, nodesY..., T[], nodesX..., T[])
  else
    FINUFFT.finufft_setpts!(forwardPlan, nodesY..., nodesX...)
  end

  FINUFFT.finufft_exec!(forwardPlan, f, fHat)

  FINUFFT.finufft_destroy!(forwardPlan)

  return fHat
end

function LinearAlgebra.mul!(f::StridedArray, pl::Adjoint{Complex{T},<:FINNUFFTPlan{T}}, fHat::AbstractArray;
         verbose=false, timing::Union{Nothing,TimingStats} = nothing) where {T}
  p = pl.parent

  D = size(p.k,1)       

  adjointPlan = FINUFFT.finufft_makeplan(3,D,1,1,p.reltol;
              nthreads = Threads.nthreads(), 
              fftw = p.fftflags)

  nodesX = ntuple(d->vec(p.k[d,:]), D)
  nodesY = ntuple(d->vec(p.y[d,:]), D)

  if D==1
    FINUFFT.finufft_setpts!(adjointPlan, nodesX..., T[], T[], nodesY..., T[], T[])
  elseif D==2
    FINUFFT.finufft_setpts!(adjointPlan, nodesX..., T[], nodesY...,T[])
  else
    FINUFFT.finufft_setpts!(adjointPlan, nodesX..., nodesY...)
  end

  FINUFFT.finufft_exec!(adjointPlan, fHat, f)

  FINUFFT.finufft_destroy!(adjointPlan)

  return f
end
