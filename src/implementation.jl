#=
Some internal documentation (especially for people familiar with the nfft)

- The window is precomputed during construction of the NFFT plan
  When performing the nfft convolution, the LUT of the window is used to
  perform linear interpolation. This approach is reasonable fast and does not
  require too much memory. There are, however alternatives known that are either
  faster or require no extra memory at all.

The non-exported functions apodization and convolve are implemented
using Cartesian macros, that may not be very readable.
This is a conscious decision where performance has outweighed readability.
More readable versions can be (and have been) written using the CartesianRange approach,
but at the time of writing this approach require *a lot* of memory.
=#

Base.@kwdef mutable struct NFFTParams{T}
  m::Int = 4
  σ::T = 2.0
  window::Symbol = :kaiser_bessel
  LUTSize::Int64 = 20000
  precompute::PrecomputeFlags = LUT
  sortNodes::Bool = false
  storeApodizationIdx::Bool = false
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
    apodizationIdx::Array{Int64,1}
    windowLUT::Vector{Vector{T}}
    windowHatInvLUT::Vector{Vector{T}}
    B::SparseMatrixCSC{T,Int64}
end

function Base.copy(p::NFFTPlan{T,D,R}) where {T,D,R}
    tmpVec = similar(p.tmpVec)
    tmpVecHat = similar(p.tmpVecHat)
    apodizationIdx = copy(p.apodizationIdx)
    windowLUT = copy(p.windowLUT)
    windowHatInvLUT = copy(p.windowHatInvLUT)
    B = copy(p.B)
    x = copy(p.x)

    FP = plan_fft!(tmpVec, p.dims; flags = p.forwardFFT.flags)
    BP = plan_bfft!(tmpVec, p.dims; flags = p.backwardFFT.flags)

    return NFFTPlan{T,D,R}(p.N, p.NOut, p.M, x, p.n, p.dims, p.params, FP, BP, tmpVec, 
                           tmpVecHat, apodizationIdx, windowLUT, windowHatInvLUT, B)
end

################
# constructor
################

function NFFTPlan(x::Matrix{T}, N::NTuple{D,Int}; dims::Union{Integer,UnitRange{Int64}}=1:D,
                 fftflags=nothing, kwargs...) where {T,D}

    params, N, NOut, M, n, dims_ = initParams(x, N, dims; kwargs...)
    
    tmpVec = Array{Complex{T},D}(undef, n)

    fftflags_ = (fftflags != nothing) ? (flags=fftflags,) : NamedTuple()
    FP = plan_fft!(tmpVec, dims_; fftflags_...)
    BP = plan_bfft!(tmpVec, dims_; fftflags_...)

    windowLUT, windowHatInvLUT, apodizationIdx, B = precomputation(x, N[dims_], n[dims_], params)
    
    U = params.storeApodizationIdx ? N : ntuple(d->0,D)
    tmpVecHat = Array{Complex{T},D}(undef, U)

    NFFTPlan(N, NOut, M, x, n, dims_, params, FP, BP, tmpVec, tmpVecHat, 
                       apodizationIdx, windowLUT, windowHatInvLUT, B)
end

function AbstractNFFTs.nodes!(p::NFFTPlan{T}, x::Matrix{T}) where {T}
    # Sort nodes in lexicographic way
    if p.params.sortNodes
        x .= sortslices(x, dims=2)
    end

    windowLUT, windowHatInvLUT, apodizationIdx, B = precomputation(x, p.N, p.n, p.params)

    p.M = size(x, 2)
    p.windowLUT = windowLUT
    p.windowHatInvLUT = windowHatInvLUT
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

function AbstractNFFTs.nfft!(p::NFFTPlan{T,D,R}, f::AbstractArray, fHat::StridedArray;
               verbose=false, timing::Union{Nothing,TimingStats} = nothing) where {T,D,R}
    consistencyCheck(p, f, fHat)

    fill!(p.tmpVec, zero(Complex{T}))
    t1 = @elapsed @inbounds apodization!(p, f, p.tmpVec)
    #if nprocs() == 1
        t2 = @elapsed p.forwardFFT * p.tmpVec # fft!(p.tmpVec) or fft!(p.tmpVec, dim)
    #else
    #    t2 = @elapsed dim(p) == 0 ? fft!(p.tmpVec) : fft!(p.tmpVec, dim(p))
    #end
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

function AbstractNFFTs.nfft_adjoint!(p::NFFTPlan, fHat::AbstractArray, f::StridedArray;
                       verbose=false, timing::Union{Nothing,TimingStats} = nothing)
    consistencyCheck(p, f, fHat)

    t1 = @elapsed @inbounds convolve_adjoint!(p, fHat, p.tmpVec)
    #if nprocs() == 1
        t2 = @elapsed p.backwardFFT * p.tmpVec # bfft!(p.tmpVec) or bfft!(p.tmpVec, dim)
    #else
    #    t2 = @elapsed dim(p) == 0 ? bfft!(p.tmpVec) : bfft!(p.tmpVec, dim(p))
    #end
    t3 = @elapsed @inbounds apodization_adjoint!(p, p.tmpVec, f)
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
