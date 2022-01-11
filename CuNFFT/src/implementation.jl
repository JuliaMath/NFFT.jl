mutable struct CuNFFTPlan{T,D} <: AbstractNFFTPlan{T,D,1} 
  N::NTuple{D,Int64}
  NOut::NTuple{1,Int64}
  M::Int64
  x::Matrix{T}
  n::NTuple{D,Int64}
  dims::UnitRange{Int64}
  params::NFFTParams{T}
  forwardFFT::CUDA.CUFFT.cCuFFTPlan{Complex{T},-1,true,D}
  backwardFFT::CUDA.CUFFT.cCuFFTPlan{Complex{T},1,true,D}
  tmpVec::CuArray{Complex{T},D}
  tmpVecHat::CuArray{Complex{T},D}
  apodizationIdx::CuArray{Int64,1}
  windowLUT::Vector{Vector{T}}
  windowHatInvLUT::CuArray{Complex{T}} # ::Vector{Vector{T}}
  B::CuSparseMatrixCSC{Complex{T}} # ::SparseMatrixCSC{T,Int64}
end

function AbstractNFFTs.plan_nfft(::Type{CuArray}, x::Matrix{T}, N::NTuple{D,Int}, rest...;
  timing::Union{Nothing,TimingStats} = nothing, kargs...) where {T,D}
  t = @elapsed begin
    p = CuNFFTPlan(x, N, rest...; kargs...)
  end
  if timing != nothing
    timing.pre = t
  end
  return p
end

function CuNFFTPlan(x::Matrix{T}, N::NTuple{D,Int}; dims::Union{Integer,UnitRange{Int64}}=1:D,
                 fftflags=nothing, kwargs...) where {T,D}

    if dims != 1:D
      error("CuNFFT does not work along directions right now!")
    end

    params, N, NOut, M, n, dims_ = NFFT.initParams(x, N, dims; kwargs...)
    params.storeApodizationIdx = true # CuNFFT only works this way
    params.precompute = NFFT.FULL # CuNFFT only works this way

    tmpVec = CuArray{Complex{T},D}(undef, n)

    #fftflags_ = (fftflags != nothing) ? (flags=fftflags,) : NamedTuple()
    #FP = plan_fft!(tmpVec, dims_; fftflags_...)
    #BP = plan_bfft!(tmpVec, dims_; fftflags_...)
    FP = plan_fft!(tmpVec, dims_)
    BP = plan_bfft!(tmpVec, dims_)

    windowLUT, windowHatInvLUT, apodizationIdx, B = NFFT.precomputation(x, N[dims_], n[dims_], params)
    
    U = params.storeApodizationIdx ? N : ntuple(d->0,D)
    tmpVecHat = CuArray{Complex{T},D}(undef, U)

    apodIdx = CuArray(apodizationIdx)
    winHatInvLUT = CuArray(Complex{T}.(windowHatInvLUT[1])) 
    B_ = CuSparseMatrixCSC(Complex{T}.(B))

    CuNFFTPlan(N, NOut, M, x, n, dims_, params, FP, BP, tmpVec, tmpVecHat, 
               apodIdx, windowLUT, winHatInvLUT, B_)
end

AbstractNFFTs.size_in(p::CuNFFTPlan) = p.N
AbstractNFFTs.size_out(p::CuNFFTPlan) = p.NOut

"""  in-place NFFT on the GPU"""
function AbstractNFFTs.nfft!(p::CuNFFTPlan{T,D}, f::CuArray, fHat::CuArray) where {T,D} 
  NFFT.consistencyCheck(p, f, fHat)

  # apodization
  fill!(p.tmpVec, zero(Complex{T}))
  p.tmpVecHat[:] .= vec(f).*p.windowHatInvLUT
  p.tmpVec[p.apodizationIdx] = p.tmpVecHat

  # FFT
  p.forwardFFT * p.tmpVec # equivalent to fft!(p.tmpVec)

  # convolution
  mul!(fHat, transpose(p.B), vec(p.tmpVec)) 

  return fHat
end

"""  in-place adjoint NFFT on the GPU"""
function AbstractNFFTs.nfft_adjoint!(p::CuNFFTPlan{T,D}, fHat::CuArray, f::CuArray) where {T,D}
  NFFT.consistencyCheck(p, f, fHat)

  # adjoint convolution
  mul!(vec(p.tmpVec), p.B, fHat)

  # FFT
  p.backwardFFT * p.tmpVec # bfft!(p.tmpVec)

  # adjoint apodization
  p.tmpVecHat[:] = p.tmpVec[p.apodizationIdx]
  f[:] .= vec(p.tmpVecHat) .* p.windowHatInvLUT
  
  return f
end
