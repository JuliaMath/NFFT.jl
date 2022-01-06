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


#=
T is the element type (Float32/Float64)
D is the number of dimensions of the input array.
R is the number of dimensions of the output array.
The transformation is done along dims.
=#
mutable struct NFFTPlan{T,D,R} <: AbstractNFFTPlan{T,D,R} 
    N::NTuple{D,Int64}
    NOut::NTuple{R,Int64}
    M::Int64
    x::Matrix{T}
    m::Int64
    σ::T
    n::NTuple{D,Int64}
    dims::UnitRange{Int64}
    dimOut::Int64
    LUTSize::Int64
    windowLUT::Vector{Vector{T}}
    windowHatInvLUT::Vector{Vector{T}}
    forwardFFT::FFTW.cFFTWPlan{Complex{T},-1,true,D,UnitRange{Int64}}
    backwardFFT::FFTW.cFFTWPlan{Complex{T},1,true,D,UnitRange{Int64}}
    tmpVec::Array{Complex{T},D}
    B::SparseMatrixCSC{T,Int64}
end

function Base.copy(p::NFFTPlan{T,D,R}) where {T,D,R}
    tmpVec = similar(p.tmpVec)
    windowLUT = copy(p.windowLUT)
    windowHatInvLUT = copy(p.windowHatInvLUT)
    B = copy(p.B)
    x = copy(p.x)

    FP = plan_fft!(tmpVec, p.dims; flags = p.forwardFFT.flags)
    BP = plan_bfft!(tmpVec, p.dims; flags = p.backwardFFT.flags)

    return NFFTPlan{T,D,R}(p.N, p.NOut, p.M, x, p.m, p.σ, p.n, p.dims, p.dimOut, p.LUTSize, windowLUT,
        windowHatInvLUT, FP, BP, tmpVec, B)
end

dims(::NFFTPlan{T,D,R}) where {T,D,R} = p.dims
dimOut(::NFFTPlan{T,D,R}) where {T,D,R} = p.dimOut
doTrafo(p::NFFTPlan{T,D}) where {T,D} = ntuple(d -> p.n[d]>p.N[d], D)

################
# constructors
################

function NFFTPlan(x::Matrix{T}, N::NTuple{D,Int}; m = 4, σ = 2.0,
                window=:kaiser_bessel, LUTSize=20000, dims::Union{Integer,UnitRange{Int64}}=1:D,
                precompute::PrecomputeFlags=LUT, sortNodes=false, kwargs...) where {D,T,R}

    # convert dims to a unit range
    dims_ = (typeof(dims) <: Integer) ? (dims:dims) : dims

    if length(dims_) != size(x,1)
        throw(ArgumentError("Nodes x have dimension $(size(x,1)) != $(length(dims_))"))
    end

    if any(isodd.(N[dims]))
      throw(ArgumentError("N = $N needs to consist of even integers along dims = $(dims)!"))
    end

    doTrafo = ntuple(d->d ∈ dims_, D)

    n = ntuple(d -> doTrafo[d] ? 
                        (ceil(Int,σ*N[d])÷2)*2 : # ensure that n is an even integer 
                         N[d], D)

    σ = n[dims_[1]] / N[dims_[1]]

    tmpVec = Array{Complex{T},D}(undef, n)

    M = size(x, 2)

    # calculate output size
    NOut = Int[]
    Mtaken = false
    for d=1:D
      if !doTrafo[d]
        push!(NOut, N[d])
      elseif !Mtaken
        push!(NOut, M)
        Mtaken = true
      end
    end
    dimOut = first(dims_)

    FP = plan_fft!(tmpVec, dims_; kwargs...)
    BP = plan_bfft!(tmpVec, dims_; kwargs...)

    # Sort nodes in lexicographic way
    if sortNodes
        x .= sortslices(x, dims=2)
    end

    windowLUT, windowHatInvLUT, B = precomputation(x, N[dims_], n[dims_], m, σ, window, LUTSize, precompute)

    NFFTPlan(N, Tuple(NOut), M, x, m, T(σ), n, dims_, dimOut, LUTSize, windowLUT, windowHatInvLUT, FP, BP, tmpVec, B)
end

function NFFTPlan!(p::AbstractNFFTPlan{T}, x::Matrix{T}, window = :kaiser_bessel; sortNodes=false) where {T}

    if isempty(p.B)
        precompute = LUT
    else
        precompute = FULL
    end

    # Sort nodes in lexicographic way
    if sortNodes
        x .= sortslices(x, dims=2)
    end

    windowLUT, windowHatInvLUT, B = precomputation(x, p.N, p.n, p.m, p.σ, window, p.LUTSize, precompute)

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
"""
        nfft!(p, f, fHat) -> fHat

Calculate the NFFT of `f` with plan `p` and store the result in `fHat`.

Both `f` and `fHat` must be complex arrays.
"""
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

"""
        nfft_adjoint!(p, fHat, f) -> f

Calculate the adjoint NFFT of `fHat` and store the result in `f`.

Both `f` and `fHat` must be complex arrays.
"""
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
