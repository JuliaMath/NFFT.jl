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
  deconvolveIdx::CuArray{Int64,1}
  windowLinInterp::Vector{T}
  windowHatInvLUT::CuArray{Complex{T}} # ::Vector{Vector{T}}
  B::CuSparseMatrixCSC{Complex{T}} # ::SparseMatrixCSC{T,Int64}
end

function AbstractNFFTs.plan_nfft(::Type{<:CuArray}, x::Matrix{T}, N::NTuple{D,Int}, rest...;
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
    params.storeDeconvolutionIdx = true # CuNFFT only works this way
    params.precompute = NFFT.FULL # CuNFFT only works this way

    tmpVec = CuArray{Complex{T},D}(undef, n)

    #fftflags_ = (fftflags != nothing) ? (flags=fftflags,) : NamedTuple()
    #FP = plan_fft!(tmpVec, dims_; fftflags_...)
    #BP = plan_bfft!(tmpVec, dims_; fftflags_...)
    FP = plan_fft!(tmpVec, dims_)
    BP = plan_bfft!(tmpVec, dims_)

    windowLinInterp, windowPolyInterp, windowHatInvLUT, deconvolveIdx, B = NFFT.precomputation(x, N[dims_], n[dims_], params)

    U = params.storeDeconvolutionIdx ? N : ntuple(d->0,D)
    tmpVecHat = CuArray{Complex{T},D}(undef, U)

    deconvIdx = CuArray(deconvolveIdx)
    winHatInvLUT = CuArray(Complex{T}.(windowHatInvLUT[1])) 
    B_ = CuSparseMatrixCSC(Complex{T}.(B))

    CuNFFTPlan{T,D}(N, NOut, M, x, n, dims_, params, FP, BP, tmpVec, tmpVecHat, 
               deconvIdx, windowLinInterp, winHatInvLUT, B_)
end

AbstractNFFTs.size_in(p::CuNFFTPlan) = p.N
AbstractNFFTs.size_out(p::CuNFFTPlan) = p.NOut

"""  in-place NFFT on the GPU"""
function LinearAlgebra.mul!(fHat::CuArray, p::CuNFFTPlan{T,D}, f::CuArray; 
                          verbose=false, timing::Union{Nothing,TimingStats} = nothing) where {T,D} 
  NFFT.consistencyCheck(p, f, fHat)

  # deconvolve
  fill!(p.tmpVec, zero(Complex{T}))
  t1 = @elapsed begin
    p.tmpVecHat[:] .= vec(f).*p.windowHatInvLUT
    p.tmpVec[p.deconvolveIdx] = p.tmpVecHat
  end

  # FFT
  t2 = @elapsed p.forwardFFT * p.tmpVec 

  # convolution
  t3 = @elapsed mul!(fHat, transpose(p.B), vec(p.tmpVec)) 

  if verbose
    @info "Timing: deconv=$t1 fft=$t2 conv=$t3"
  end
  if timing != nothing
    timing.conv = t3
    timing.fft = t2
    timing.deconv = t1
  end

  return fHat
end

"""  in-place adjoint NFFT on the GPU"""
function LinearAlgebra.mul!(f::CuArray, pl::Adjoint{Complex{T},<:CuNFFTPlan{T,D}}, fHat::CuArray;
                       verbose=false, timing::Union{Nothing,TimingStats} = nothing) where {T,D}
  p = pl.parent
  
  NFFT.consistencyCheck(p, f, fHat)

  # adjoint convolution
  t1 = @elapsed mul!(vec(p.tmpVec), p.B, fHat)

  # FFT
  t2 = @elapsed p.backwardFFT * p.tmpVec 

  # adjoint deconvolve
  t3 = @elapsed begin
    p.tmpVecHat[:] = p.tmpVec[p.deconvolveIdx]
    f[:] .= vec(p.tmpVecHat) .* p.windowHatInvLUT
  end

  if verbose
    @info "Timing: conv=$t1 fft=$t2 deconv=$t3"
  end
  if timing != nothing
    timing.conv_adjoint = t1
    timing.fft_adjoint = t2
    timing.deconv_adjoint = t3
  end

  return f
end
