import NFFT3

mutable struct NFFT3Plan{D} <: AbstractNFFTPlan{Float64,D,1} 
  parent::NFFT3.NFFT{D}
end

dims(p::NFFT3Plan) = Int.(reverse(p.parent.N))
dimOut(p::NFFT3Plan) = Int(p.parent.M)

################
# constructors
################

function NFFT3Plan(x::Matrix{T}, N::NTuple{D,Int}; m = 4, σ = 2.0,
              dims::Union{Integer,UnitRange{Int64}}=1:D,
              precompute::PrecomputeFlags=LUT, sortNodes=false, 
              flags=UInt32(NFFT3.FFTW_ESTIMATE | NFFT3.FFTW_DESTROY_INPUT), 
              kwargs...) where {D,T}

  if dims != 1:D
    error("NFFT3 does not support directional plans!")
  end

  prePsi = (precompute == AbstractNFFTs.LUT) ? NFFT3.PRE_LIN_PSI : NFFT3.PRE_FULL_PSI
  sortN = sortNodes ? NFFT3.NFCT_SORT_NODES : UInt32(0)

  f1 = UInt32(
    NFFT3.PRE_PHI_HUT |
    prePsi |
    NFFT3.MALLOC_X |
    NFFT3.MALLOC_F_HAT |
    NFFT3.MALLOC_F |
    NFFT3.FFTW_INIT |
    NFFT3.FFT_OUT_OF_PLACE |
    sortN |
    NFFT3.NFCT_OMP_BLOCKWISE_ADJOINT
     )

  f2 = UInt32(flags)

  n = ntuple(d -> (ceil(Int,σ*N[d])÷2)*2, D) # ensure that n is an even integer 
  σ = n[1] / N[1]

  parent = NFFT3.NFFT(Int32.(reverse(N)), size(x,2), Int32.(n), Int32(m), f1, f2)

  xx = Float64.(reverse(x, dims=1))
  parent.x = D == 1 ? vec(xx) : xx

  return NFFT3Plan(parent)
end


function Base.show(io::IO, p::NFFT3Plan{D}) where {D}
  print(io, "NFFT3Plan")
end

AbstractNFFTs.size_in(p::NFFT3Plan) = reverse(Int.(p.parent.N))
AbstractNFFTs.size_out(p::NFFT3Plan) = (Int(p.parent.M),)

function AbstractNFFTs.nfft!(p::NFFT3Plan{D}, f::AbstractArray, fHat::StridedArray;
             verbose=false, timing::Union{Nothing,TimingStats} = nothing) where {D}
  #consistencyCheck(p, f, fHat)

  p.parent.fhat = vec(f)
  NFFT3.nfft_trafo(p.parent)
  fHat[:] .= p.parent.f  

  return fHat
end

function AbstractNFFTs.nfft_adjoint!(p::NFFT3Plan, fHat::AbstractArray, f::StridedArray;
                     verbose=false, timing::Union{Nothing,TimingStats} = nothing)
  #consistencyCheck(p, f, fHat)

  p.parent.f = vec(fHat)
  tadjoint = fApprox = NFFT3.nfft_adjoint(p.parent)
  f[:] .= p.parent.fhat

  return f
end
