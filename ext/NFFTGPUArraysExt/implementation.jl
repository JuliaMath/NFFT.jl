mutable struct GPU_NFFTPlan{T,D, arrTc <: AbstractGPUArray{Complex{T}, D}, vecI <: AbstractGPUVector{Int32}, FP, BP, INV, SM} <: AbstractNFFTPlan{T,D,1} 
  N::NTuple{D,Int64}
  NOut::NTuple{1,Int64}
  J::Int64
  k::Matrix{T}
  Ñ::NTuple{D,Int64}
  dims::UnitRange{Int64}
  params::NFFTParams{T}
  forwardFFT::FP
  backwardFFT::BP
  tmpVec::arrTc
  tmpVecHat::arrTc
  deconvolveIdx::vecI
  windowLinInterp::Vector{T}
  windowHatInvLUT::INV
  B::SM
end

# Atm initParams is not supported for k != Array, so in the case of inferred arr from k::GPUArray we need to convert k to Array
AbstractNFFTs.plan_nfft(::NFFTBackend, arr::Type{<:AbstractGPUArray}, k::AbstractMatrix, args...; kwargs...) = AbstractNFFTs.plan_nfft(arr, Array(k), args...; kwargs...)
function AbstractNFFTs.plan_nfft(::NFFTBackend, arr::Type{<:AbstractGPUArray}, k::Matrix{T}, N::NTuple{D,Int}, rest...;
  timing::Union{Nothing,TimingStats} = nothing, kargs...) where {T,D}
  t = @elapsed begin
    p = GPU_NFFTPlan(arr, k, N, rest...; kargs...)
  end
  if timing != nothing
    timing.pre = t
  end
  return p
end

function GPU_NFFTPlan(arr, k::Matrix{T}, N::NTuple{D,Int}; dims::Union{Integer,UnitRange{Int64}}=1:D,
                 fftflags=nothing, kwargs...) where {T,D}

    if dims != 1:D
      error("GPU NFFT does not work along directions right now!")
    end

    params, N, NOut, J, Ñ, dims_ = NFFT.initParams(k, N, dims; kwargs...)
    params.storeDeconvolutionIdx = true # GPU_NFFT only works this way
    params.precompute = NFFT.FULL # GPU_NFFT only works this way

    tmpVec = adapt(arr, zeros(Complex{T}, Ñ))

    FP = plan_fft!(tmpVec, dims_)
    BP = plan_bfft!(tmpVec, dims_)

    windowLinInterp, windowPolyInterp, windowHatInvLUT, deconvolveIdx, B = NFFT.precomputation(k, N[dims_], Ñ[dims_], params)

    U = params.storeDeconvolutionIdx ? N : ntuple(d->0,D)
    tmpVecHat = adapt(arr, zeros(Complex{T}, U))

    deconvIdx = Int32.(adapt(arr, (deconvolveIdx)))
    winHatInvLUT = Complex{T}.(adapt(arr, (windowHatInvLUT[1]))) 
    B_ = Complex{T}.(adapt(arr, (B))) # Bit hacky

    GPU_NFFTPlan{T,D, typeof(tmpVec), typeof(deconvIdx), typeof(FP), typeof(BP), typeof(winHatInvLUT), typeof(B_)}(N, NOut, J, k, Ñ, dims_, params, FP, BP, tmpVec, tmpVecHat, 
               deconvIdx, windowLinInterp, winHatInvLUT, B_)
end

AbstractNFFTs.size_in(p::GPU_NFFTPlan) = p.N
AbstractNFFTs.size_out(p::GPU_NFFTPlan) = p.NOut


function AbstractNFFTs.convolve!(p::GPU_NFFTPlan{T,D, arrTc}, g::arrTc, fHat::arrH) where {D,T,arr<: AbstractGPUArray, arrTc <: arr, arrH <: arr}
  mul!(fHat, transpose(p.B), vec(g)) 
  return
end

function AbstractNFFTs.convolve_transpose!(p::GPU_NFFTPlan{T,D, arrTc}, fHat::arrH, g::arrTc) where {D,T,arr<: AbstractGPUArray, arrTc <: arr, arrH <: arr}
  mul!(vec(g), p.B, fHat)
  return
end

function AbstractNFFTs.deconvolve!(p::GPU_NFFTPlan{T,D, arrTc}, f::arrF, g::arrTc) where {D,T,arr<: AbstractGPUArray, arrTc <: arr, arrF <: arr}
  p.tmpVecHat[:] .= vec(f) .* p.windowHatInvLUT
  g[p.deconvolveIdx] = p.tmpVecHat
  return
end

function AbstractNFFTs.deconvolve_transpose!(p::GPU_NFFTPlan{T,D, arrTc}, g::arrTc, f::arrF) where {D,T,arr<: AbstractGPUArray, arrTc <: arr, arrF <: arr}
  p.tmpVecHat[:] .= broadcast(p.deconvolveIdx) do idx
    g[idx]
  end
  f[:] .= vec(p.tmpVecHat) .* p.windowHatInvLUT
  return
end

"""  in-place NFFT on the GPU"""
function LinearAlgebra.mul!(fHat::arrH, p::GPU_NFFTPlan{T,D, arrT}, f::arrF; 
                          verbose=false, timing::Union{Nothing,TimingStats} = nothing) where {T,D,arr<: AbstractGPUArray, arrT <: arr, arrH <: arr, arrF <: arr} 
    NFFT.consistencyCheck(p, f, fHat)

    fill!(p.tmpVec, zero(Complex{T}))
    t1 = @elapsed @inbounds deconvolve!(p, f, p.tmpVec)
    t2 = @elapsed p.forwardFFT * p.tmpVec
    t3 = @elapsed @inbounds convolve!(p, p.tmpVec, fHat)
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
function LinearAlgebra.mul!(f::arrF, pl::Adjoint{Complex{T},<:GPU_NFFTPlan{T,D, arrT}}, fHat::arrH;
                       verbose=false, timing::Union{Nothing,TimingStats} = nothing) where {T,D,arr<: AbstractGPUArray, arrT <: arr, arrH <: arr, arrF <: arr}
    p = pl.parent
    NFFT.consistencyCheck(p, f, fHat)

    t1 = @elapsed @inbounds convolve_transpose!(p, fHat, p.tmpVec)
    t2 = @elapsed p.backwardFFT * p.tmpVec
    t3 = @elapsed @inbounds deconvolve_transpose!(p, p.tmpVec, f)
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

