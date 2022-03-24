

Base.@kwdef mutable struct NFFTParams{T}
  m::Int = 4
  σ::T = 2.0
  reltol::T = 1e-7
  window::Symbol = :kaiser_bessel
  LUTSize::Int64 = 0
  precompute::PrecomputeFlags = POLYNOMIAL
  sortNodes::Bool = false
  storeApodizationIdx::Bool = false
  blocking::Bool = true
end

mutable struct NFFTPlan{T,D,R} <: AbstractNFFTPlan{T,D,R}
    N::NTuple{D,Int64}
    NOut::NTuple{R,Int64}
    M::Int64
    x::Matrix{T}
    n::NTuple{D,Int64}
    dims::UnitRange{Int64}
    params::NFFTParams{T}
    forwardFFT::FFTW.cFFTWPlan{Complex{T},-1,true,D,UnitRange{Int64}}
    backwardFFT::FFTW.cFFTWPlan{Complex{T},1,true,D,UnitRange{Int64}}
    tmpVec::Array{Complex{T},D}
    tmpVecHat::Array{Complex{T},D}
    # Caches for deconvolve
    deconvolveIdx::Array{Int64,1}
    windowHatInvLUT::Vector{Vector{T}}
    # Cache for precompute = LUT
    windowLinInterp::Vector{T}
    # Cache for precompute = POLYNOMIAL
    windowPolyInterp::Matrix{T}
    # Caches for blocking
    blocks::Array{Array{Complex{T},D},D}
    nodesInBlock::Array{Vector{Int64},D}
    blockOffsets::Array{NTuple{D,Int64},D}
    idxInBlock::Array{Matrix{Tuple{Int,T}},D}
    windowTensor::Array{Array{T,3},D}
    # Cache for precompute = FULL
    B::SparseMatrixCSC{T,Int64}
end

function Base.copy(p::NFFTPlan{T,D,R}) where {T,D,R}
    tmpVec = similar(p.tmpVec)
    tmpVecHat = similar(p.tmpVecHat)
    deconvolveIdx = copy(p.deconvolveIdx)
    windowLinInterp = copy(p.windowLinInterp)
    windowPolyInterp = copy(p.windowPolyInterp)
    windowHatInvLUT = copy(p.windowHatInvLUT)
    B = copy(p.B)
    blocks = deepcopy(p.blocks)
    nodesInBlock = deepcopy(p.nodesInBlock)
    blockOffsets = copy(p.blockOffsets)
    idxInBlock = copy(p.idxInBlock)
    windowTensor = copy(p.windowTensor)
    x = p.x

    FP = plan_fft!(tmpVec, p.dims; flags = p.forwardFFT.flags)
    BP = plan_bfft!(tmpVec, p.dims; flags = p.backwardFFT.flags)

    return NFFTPlan{T,D,R}(p.N, p.NOut, p.M, x, p.n, p.dims, p.params, FP, BP, tmpVec,
                           tmpVecHat, deconvolveIdx, windowHatInvLUT, windowLinInterp, windowPolyInterp,
                           blocks, nodesInBlock, blockOffsets, idxInBlock, windowTensor, B)
end


################
# constructor
################

function NFFTPlan(x::Matrix{T}, N::NTuple{D,Int}; dims::Union{Integer,UnitRange{Int64}}=1:D,
                 fftflags=nothing, kwargs...) where {T,D}

    checkNodes(x)

    params, N, NOut, M, n, dims_ = initParams(x, N, dims; kwargs...)

    if length(NOut) > 1
      params.precompute = LINEAR
    end

    tmpVec = Array{Complex{T},D}(undef, n)

    fftflags_ = (fftflags != nothing) ? (flags=fftflags,) : NamedTuple()
    FP = plan_fft!(tmpVec, dims_; fftflags_...)
    BP = plan_bfft!(tmpVec, dims_; fftflags_...)

    calcBlocks = (params.precompute == LINEAR ||
                  params.precompute == TENSOR ||
                  params.precompute == POLYNOMIAL ) &&
                     params.blocking && length(dims_) == D

    blocks, nodesInBlocks, blockOffsets, idxInBlock, windowTensor = precomputeBlocks(x, n, params, calcBlocks)

    windowLinInterp, windowPolyInterp, windowHatInvLUT, deconvolveIdx, B =
            precomputation(x, N[dims_], n[dims_], params)

    U = params.storeApodizationIdx ? N : ntuple(d->0,D)
    tmpVecHat = Array{Complex{T},D}(undef, U)

    NFFTPlan(N, NOut, M, x, n, dims_, params, FP, BP, tmpVec, tmpVecHat,
                       deconvolveIdx, windowHatInvLUT, windowLinInterp, windowPolyInterp,
                       blocks, nodesInBlocks, blockOffsets, idxInBlock, windowTensor, B)
end

function AbstractNFFTs.nodes!(p::NFFTPlan{T}, x::Matrix{T}) where {T}
    checkNodes(x)

    # Sort nodes in lexicographic way
    if p.params.sortNodes
        x .= sortslices(x, dims=2)
    end

    calcBlocks = (p.params.precompute == LINEAR ||
                  p.params.precompute == TENSOR ||
                  p.params.precompute == POLYNOMIAL ) &&
                     p.params.blocking && length(p.dims) == length(p.N)

    blocks, nodesInBlocks, blockOffsets, idxInBlock, windowTensor = precomputeBlocks(x, p.n, p.params, calcBlocks)

    windowLinInterp, windowPolyInterp, windowHatInvLUT, deconvolveIdx, B =
       precomputation(x, p.N, p.n, p.params)

    p.blocks = blocks
    p.nodesInBlock = nodesInBlocks
    p.blockOffsets = blockOffsets
    p.idxInBlock = idxInBlock
    p.windowTensor = windowTensor

    p.M = size(x, 2)
    p.windowLinInterp = windowLinInterp
    p.windowPolyInterp = windowPolyInterp
    p.windowHatInvLUT = windowHatInvLUT
    p.deconvolveIdx = deconvolveIdx
    p.B = B
    p.x = x

    return p
end

function Base.show(io::IO, p::NFFTPlan{T,D,R}) where {T,D,R}
    print(io, "NFFTPlan with ", p.M, " sampling points for an input array of size",
           p.N, " and an output array of size", p.NOut, " with dims ", p.dims)
end

AbstractNFFTs.size_in(p::NFFTPlan) = p.N
AbstractNFFTs.size_out(p::NFFTPlan) = p.NOut

################
# nfft functions
################

function LinearAlgebra.mul!(fHat::StridedArray, p::NFFTPlan{T,D,R}, f::AbstractArray;
               verbose=false, timing::Union{Nothing,TimingStats} = nothing) where {T,D,R}
    consistencyCheck(p, f, fHat)

    fill!(p.tmpVec, zero(Complex{T}))
    t1 = @elapsed @inbounds deconvolve!(p, f, p.tmpVec)
    t2 = @elapsed p.forwardFFT * p.tmpVec
    t3 = @elapsed @inbounds convolve!(p, p.tmpVec, fHat)
    if verbose
        @info "Timing: apod=$t1 fft=$t2 conv=$t3"
    end
    if timing != nothing
      timing.conv = t3
      timing.fft = t2
      timing.apod = t1
    end
    return fHat
end



function LinearAlgebra.mul!(f::StridedArray, pl::Adjoint{Complex{T},<:NFFTPlan{T}}, fHat::AbstractArray;
                       verbose=false, timing::Union{Nothing,TimingStats} = nothing) where {T}
    p = pl.parent
    consistencyCheck(p, f, fHat)

    t1 = @elapsed @inbounds convolve_transpose!(p, fHat, p.tmpVec)
    t2 = @elapsed p.backwardFFT * p.tmpVec
    t3 = @elapsed @inbounds deconvolve_transpose!(p, p.tmpVec, f)
    if verbose
        @info "Timing: conv=$t1 fft=$t2 apod=$t3"
    end
    if timing != nothing
      timing.conv_adjoint = t1
      timing.fft_adjoint = t2
      timing.apod_adjoint = t3
    end
    return f
end
