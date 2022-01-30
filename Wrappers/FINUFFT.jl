using LinearAlgebra
import FINUFFT

mutable struct FINUFFTPlan{T,D} <: AbstractNFFTPlan{T,D,1} 
  N::NTuple{D,Int64}
  M::Int64
  x::Matrix{T}
  m::Int
  σ::T
  reltol::T
  fftflags::UInt32
  #forwardPlan::FINUFFT.finufft_plan{T}
  #adjointPlan::FINUFFT.finufft_plan{T}
end


################
# constructors
################

function FINUFFTPlan(x::Matrix{T}, N::NTuple{D,Int}; 
              dims::Union{Integer,UnitRange{Int64}}=1:D,
              fftflags=UInt32(NFFT.FFTW.ESTIMATE), 
              kargs...) where {D,T}

  if dims != 1:D
  error("FINUFFT directional plans not yet implemented!")
  end

  M = size(x,2)

  m, σ, reltol = accuracyParams(; kargs...)

  reltol = max(reltol, 1.0e-15)

  #forwardPlan = FINUFFT.finufft_makeplan(2,collect(N),-1,1,reltol; 
  #                              nthreads = Threads.nthreads(), 
  #                              fftw = fftflags)

  #adjointPlan = FINUFFT.finufft_makeplan(1,collect(N),1,1,reltol;
  #                              nthreads = Threads.nthreads(), 
  #                              fftw = fftflags)

  x_ = 2π * x 
  #nodes = ntuple(d->vec(x_[d,:]), D)
  #x__ = 2π * x 
  #nodes_ = ntuple(d->vec(x__[d,:]), D)
  #FINUFFT.finufft_setpts!(forwardPlan, nodes...)
  #FINUFFT.finufft_setpts!(adjointPlan, nodes_...)

  p = FINUFFTPlan(N, M, x_, m, T(σ), reltol, fftflags) #, forwardPlan, adjointPlan)



  #finalizer(p -> begin
  #  FINUFFT.finufft_destroy!(p.forwardPlan)
  #  FINUFFT.finufft_destroy!(p.adjointPlan)
  #end, p)

  return p
end


function Base.show(io::IO, p::FINUFFTPlan)
print(io, "FINUFFTPlan")
end

AbstractNFFTs.size_in(p::FINUFFTPlan) = Int.(p.N)
AbstractNFFTs.size_out(p::FINUFFTPlan) = (Int(p.M),)


function LinearAlgebra.mul!(fHat::StridedArray, p::FINUFFTPlan{T,D}, f::AbstractArray;
             verbose=false, timing::Union{Nothing,TimingStats} = nothing) where {T,D}

  forwardPlan = FINUFFT.finufft_makeplan(2,collect(p.N),-1,1,p.reltol;
                                        nthreads = Threads.nthreads(), 
                                        fftw = p.fftflags, upsampfac=2.0, debug=0)

  nodes = ntuple(d->vec(p.x[d,:]), D)
  FINUFFT.finufft_setpts!(forwardPlan, nodes...)
  
  FINUFFT.finufft_exec!(forwardPlan, f, fHat)

  FINUFFT.finufft_destroy!(forwardPlan)

  return fHat
end

function LinearAlgebra.mul!(f::StridedArray, pl::Adjoint{Complex{T},<:FINUFFTPlan{T,D}}, fHat::AbstractArray;
                     verbose=false, timing::Union{Nothing,TimingStats} = nothing) where {T,D}
  p = pl.parent

  adjointPlan = FINUFFT.finufft_makeplan(1,collect(p.N), 1, 1, p.reltol; 
                                nthreads = Threads.nthreads(), 
                                fftw = p.fftflags, upsampfac=2.0, debug=0)

  nodes = ntuple(d->vec(p.x[d,:]), D)
  FINUFFT.finufft_setpts!(adjointPlan, nodes...)
  
  FINUFFT.finufft_exec!(adjointPlan, fHat, f)

  FINUFFT.finufft_destroy!(adjointPlan)

  return f
end






mutable struct FINNUFFTPlan{T} <: AbstractNNFFTPlan{T,1,1} 
  N::Int64
  M::Int64
  x::Matrix{T}
  y::Matrix{T}
  m::Int
  σ::T
  reltol::T
  fftflags::UInt32
end

function FINNUFFTPlan(x::Matrix{T}, y::Matrix{T}; 
  fftflags=UInt32(NFFT.FFTW.ESTIMATE), 
  kargs...) where {T}

  N = size(y,2)
  M = size(x,2)

  m, σ, reltol = accuracyParams(; kargs...)

  reltol = max(reltol, 1.0e-15)

  x_ = 2π * x 
  y_ = y 

  p = FINNUFFTPlan(N, M, x_, y_, m, T(σ), reltol, fftflags)

  return p
end


function Base.show(io::IO, p::FINNUFFTPlan)
  print(io, "FINNUFFTPlan")
end

AbstractNFFTs.size_in(p::FINNUFFTPlan) = (Int.(p.N),)
AbstractNFFTs.size_out(p::FINNUFFTPlan) = (Int(p.M),)

function LinearAlgebra.mul!(fHat::StridedArray, p::FINNUFFTPlan{T}, f::AbstractArray;
 verbose=false, timing::Union{Nothing,TimingStats} = nothing) where {T}

  D = size(p.x,1)

  forwardPlan = FINUFFT.finufft_makeplan(3,D,-1,1,p.reltol;
                              nthreads = Threads.nthreads(), 
                              fftw = p.fftflags)

  nodesX = ntuple(d->vec(p.x[d,:]), D)
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

  D = size(p.x,1)       

  adjointPlan = FINUFFT.finufft_makeplan(3,D,1,1,p.reltol;
              nthreads = Threads.nthreads(), 
              fftw = p.fftflags)

  nodesX = ntuple(d->vec(p.x[d,:]), D)
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
