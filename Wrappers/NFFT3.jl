using LinearAlgebra
import NFFT3

mutable struct NFFT3Plan{T,D} <: AbstractNFFTPlan{T,D,1} 
  parent::NFFT3.NFFT{D}
end

mutable struct NFCT3Plan{T,D} <: AbstractNFCTPlan{T,D,1} 
  parent::NFFT3.NFCT{D}
end

mutable struct NFST3Plan{T,D} <: AbstractNFSTPlan{T,D,1} 
  parent::NFFT3.NFST{D}
end

################
# constructors
################

function NFFT3Plan(k::Matrix{T}, N::NTuple{D,Int}; 
              dims::Union{Integer,UnitRange{Int64}}=1:D,
              precompute::PrecomputeFlags=TENSOR, sortNodes=false, 
              fftflags=UInt32(NFFT3.FFTW_ESTIMATE), 
              kwargs...) where {D,T}

  if dims != 1:D
    error("NFFT3 does not support directional plans!")
  end

  m, σ, reltol = accuracyParams(; kwargs...)

  if precompute == AbstractNFFTs.LINEAR
    prePsi =  NFFT3.PRE_LIN_PSI 
  elseif precompute == AbstractNFFTs.FULL
    prePsi = NFFT3.PRE_FULL_PSI
  elseif precompute == AbstractNFFTs.TENSOR
    prePsi = NFFT3.PRE_PSI
  else
    error("Precompute $(precompute) not supported by NFFT3!")
  end
  sortN = sortNodes ? NFFT3.NFCT_SORT_NODES : UInt32(0)

  f1 = UInt32(
    NFFT3.PRE_PHI_HUT |
    prePsi |
    # NFFT3.MALLOC_X |
    # NFFT3.MALLOC_F_HAT |
    # NFFT3.MALLOC_F |
    NFFT3.FFTW_INIT |
    NFFT3.FFT_OUT_OF_PLACE |
    sortN |
    NFFT3.NFCT_OMP_BLOCKWISE_ADJOINT
     )

  f2 = UInt32(fftflags | NFFT3.FFTW_DESTROY_INPUT)

  Ñ = ntuple(d -> (ceil(Int,σ*N[d])÷2)*2, D) # ensure that n is an even integer 

  parent = NFFT3.NFFT(Int32.(reverse(N)), size(k,2), Int32.(reverse(Ñ)), Int32(m), f1, f2)

  xx = Float64.(reverse(k, dims=1))
  parent.x = D == 1 ? vec(xx) : xx

  return NFFT3Plan{T,D}(parent)
end

function NFCT3Plan(k::Matrix{T}, N::NTuple{D,Int}; 
              dims::Union{Integer,UnitRange{Int64}}=1:D,
              precompute::PrecomputeFlags=TENSOR, sortNodes=false, 
              fftflags=UInt32(NFFT3.FFTW_ESTIMATE), 
              kwargs...) where {D,T}

  if dims != 1:D
    error("NFFT3 does not support directional plans!")
  end

  m, σ, reltol = accuracyParams(; kwargs...)

  if precompute == AbstractNFFTs.LINEAR
    prePsi =  NFFT3.PRE_LIN_PSI 
  elseif precompute == AbstractNFFTs.FULL
    prePsi = NFFT3.PRE_FULL_PSI
  elseif precompute == AbstractNFFTs.TENSOR
    prePsi = NFFT3.PRE_PSI
  else
    error("Precompute $(precompute) not supported by NFFT3!")
  end
  sortN = sortNodes ? NFFT3.NFCT_SORT_NODES : UInt32(0)

  f1 = UInt32(
    NFFT3.PRE_PHI_HUT |
    prePsi |
    # NFFT3.MALLOC_X |
    # NFFT3.MALLOC_F_HAT |
    # NFFT3.MALLOC_F |
    NFFT3.FFTW_INIT |
    NFFT3.FFT_OUT_OF_PLACE |
    sortN |
    NFFT3.NFCT_OMP_BLOCKWISE_ADJOINT
     )

  f2 = UInt32(fftflags | NFFT3.FFTW_DESTROY_INPUT)

  Ñ = ntuple(d -> (ceil(Int,σ*N[d])÷2)*2, D) # ensure that n is an even integer 

  parent = NFFT3.NFCT(Int32.(reverse(N)), size(k,2), Int32.(reverse(Ñ)), Int32(m), f1, f2)

  xx = Float64.(reverse(k, dims=1))
  parent.x = D == 1 ? vec(xx) : xx

  return NFCT3Plan{T,D}(parent)
end

function NFST3Plan(k::Matrix{T}, N::NTuple{D,Int}; 
              dims::Union{Integer,UnitRange{Int64}}=1:D,
              precompute::PrecomputeFlags=TENSOR, sortNodes=false, 
              fftflags=UInt32(NFFT3.FFTW_ESTIMATE), 
              kwargs...) where {D,T}

  if dims != 1:D
    error("NFFT3 does not support directional plans!")
  end

  m, σ, reltol = accuracyParams(; kwargs...)

  if precompute == AbstractNFFTs.LINEAR
    prePsi =  NFFT3.PRE_LIN_PSI 
  elseif precompute == AbstractNFFTs.FULL
    prePsi = NFFT3.PRE_FULL_PSI
  elseif precompute == AbstractNFFTs.TENSOR
    prePsi = NFFT3.PRE_PSI
  else
    error("Precompute $(precompute) not supported by NFFT3!")
  end
  sortN = sortNodes ? NFFT3.NFCT_SORT_NODES : UInt32(0)

  f1 = UInt32(
    NFFT3.PRE_PHI_HUT |
    prePsi |
    # NFFT3.MALLOC_X |
    # NFFT3.MALLOC_F_HAT |
    # NFFT3.MALLOC_F |
    NFFT3.FFTW_INIT |
    NFFT3.FFT_OUT_OF_PLACE |
    sortN |
    NFFT3.NFCT_OMP_BLOCKWISE_ADJOINT
     )

  f2 = UInt32(fftflags | NFFT3.FFTW_DESTROY_INPUT)

  Ñ = ntuple(d -> (ceil(Int,σ*N[d])÷2)*2, D) # ensure that n is an even integer 

  parent = NFFT3.NFST(Int32.(reverse(N)), size(k,2), Int32.(reverse(Ñ)), Int32(m), f1, f2)

  xx = Float64.(reverse(k, dims=1))

  parent.x = D == 1 ? vec(xx) : xx

  return NFST3Plan{T,D}(parent)
end

function Base.show(io::IO, p::NFFT3Plan{D}) where {D}
  print(io, "NFFT3Plan")
end

function Base.show(io::IO, p::NFCT3Plan{D}) where {D}
  print(io, "NFCT3Plan")
end

function Base.show(io::IO, p::NFST3Plan{D}) where {D}
  print(io, "NFST3Plan")
end

AbstractNFFTs.size_in(p::NFFT3Plan) = reverse(Int.(p.parent.N))
AbstractNFFTs.size_out(p::NFFT3Plan) = (Int(p.parent.M),)

AbstractNFFTs.size_in(p::NFCT3Plan) = reverse(Int.(p.parent.N))
AbstractNFFTs.size_out(p::NFCT3Plan) = (Int(p.parent.M),)

AbstractNFFTs.size_in(p::NFST3Plan) = reverse(Int.(p.parent.N .- 1))
AbstractNFFTs.size_out(p::NFST3Plan) = (Int(p.parent.M),)

function AbstractNFFTs.plan_nfft(::Type{<:Array}, k::Matrix{T}, N::NTuple{D,Int}, rest...;
                   timing::Union{Nothing,TimingStats} = nothing, kargs...) where {T,D}
  t = @elapsed begin
    p = NFFT3Plan(k, N, rest...; kargs...)
  end
  if timing != nothing
    timing.pre = t
  end
  return p
end

function AbstractNFFTs.plan_nfct(::Type{<:Array}, k::Matrix{T}, N::NTuple{D,Int}, rest...;
                   timing::Union{Nothing,TimingStats} = nothing, kargs...) where {T,D}
  t = @elapsed begin
    p = NFCT3Plan(k, N, rest...; kargs...)
  end
  if timing != nothing
    timing.pre = t
  end
  return p
end

function AbstractNFFTs.plan_nfst(::Type{<:Array}, k::Matrix{T}, N::NTuple{D,Int}, rest...;
                   timing::Union{Nothing,TimingStats} = nothing, kargs...) where {T,D}
  t = @elapsed begin
    p = NFST3Plan(k, N, rest...; kargs...)
  end
  if timing != nothing
    timing.pre = t
  end
  return p
end

function LinearAlgebra.mul!(fHat::StridedArray, p::NFFT3Plan{D}, f::AbstractArray;
             verbose=false, timing::Union{Nothing,TimingStats} = nothing) where {D}
  #consistencyCheck(p, f, fHat)

  p.parent.fhat = vec(f)
  p.parent.f = vec(fHat)
  NFFT3.nfft_trafo(p.parent)
  fHat[:] .= p.parent.f  

  return fHat
end

function LinearAlgebra.mul!(f::StridedArray, pl::Adjoint{Complex{T},<:NFFT3Plan{T}}, fHat::AbstractArray;
                     verbose=false, timing::Union{Nothing,TimingStats} = nothing) where T
  #consistencyCheck(p, f, fHat)
  p = pl.parent

  p.parent.f = vec(fHat)
  p.parent.fhat = vec(f)
  tadjoint = fApprox = NFFT3.nfft_adjoint(p.parent)
  f[:] .= p.parent.fhat

  return f
end

function LinearAlgebra.mul!(fHat::StridedArray, p::NFCT3Plan{D}, f::AbstractArray;
             verbose=false, timing::Union{Nothing,TimingStats} = nothing) where {D}

  p.parent.fhat = vec(f)
  p.parent.f = vec(fHat)
  NFFT3.nfct_trafo(p.parent)
  fHat[:] .= p.parent.f  

  return fHat
end

function LinearAlgebra.mul!(f::StridedArray, pl::Transpose{T,<:NFCT3Plan{T}}, fHat::AbstractArray;
                     verbose=false, timing::Union{Nothing,TimingStats} = nothing) where T
  p = pl.parent

  p.parent.f = vec(fHat)
  p.parent.fhat = vec(f)
  tadjoint = fApprox = NFFT3.nfct_adjoint(p.parent)
  f[:] .= p.parent.fhat

  return f
end


function LinearAlgebra.mul!(fHat::StridedArray, p::NFST3Plan{D}, f::AbstractArray;
             verbose=false, timing::Union{Nothing,TimingStats} = nothing) where {D}

  p.parent.fhat = vec(f)
  p.parent.f = vec(fHat)
  NFFT3.nfst_trafo(p.parent)
  fHat[:] .= p.parent.f  

  return fHat
end

function LinearAlgebra.mul!(f::StridedArray, pl::Transpose{T,<:NFST3Plan{T}}, fHat::AbstractArray;
                     verbose=false, timing::Union{Nothing,TimingStats} = nothing) where T
  p = pl.parent

  p.parent.f = vec(fHat)
  p.parent.fhat = vec(f)
  tadjoint = fApprox = NFFT3.nfst_adjoint(p.parent)
  f[:] .= p.parent.fhat

  return f
end

