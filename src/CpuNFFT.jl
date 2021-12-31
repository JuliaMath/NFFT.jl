using Polyester

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
D is the number of dimensions of the array to be transformed.
DIM is the dimension along which the array is transformed.
DIM == 0 is the ordinary NFFT, i.e., where all dimensions are transformed.
DIM is a type parameter since it allows the @generated macro to
compile more efficient methods.
=#
mutable struct NFFTPlan{D,DIM,T} <: AbstractNFFTPlan{D,DIM,T}
    N::NTuple{D,Int64}
    M::Int64
    x::Matrix{T}
    m::Int64
    sigma::T
    n::NTuple{D,Int64}
    K::Int64
    windowLUT::Vector{Vector{T}}
    windowHatInvLUT::Vector{Vector{T}}
    forwardFFT::FFTW.cFFTWPlan{Complex{T},-1,true,D}
    backwardFFT::FFTW.cFFTWPlan{Complex{T},1,true,D}
    tmpVec::Array{Complex{T},D}
    B::SparseMatrixCSC{T,Int64}
end

function Base.copy(p::NFFTPlan{D,0,T}) where {D,T}
    tmpVec = similar(p.tmpVec)
    windowLUT = copy(p.windowLUT)
    windowHatInvLUT = copy(p.windowHatInvLUT)
    B = copy(p.B)
    x = copy(p.x)

    FP = plan_fft!(tmpVec; flags = p.forwardFFT.flags)
    BP = plan_bfft!(tmpVec; flags = p.backwardFFT.flags)

    return NFFTPlan{D,0,T}(p.N, p.M, x, p.m, p.sigma, p.n, p.K, windowLUT,
        windowHatInvLUT, FP, BP, tmpVec, B)
end

function Base.copy(p::NFFTPlan{D,DIM,T}) where {D,DIM,T}
    tmpVec = similar(p.tmpVec)
    windowLUT = copy(p.windowLUT)
    windowHatInvLUT = copy(p.windowHatInvLUT)
    B = copy(p.B)
    x = copy(p.x)

    FP = plan_fft!(tmpVec, DIM; flags = p.forwardFFT.flags)
    BP = plan_bfft!(tmpVec, DIM; flags = p.backwardFFT.flags)

    return NFFTPlan{D,DIM,T}(p.N, p.M, x, p.m, p.sigma, p.n, p.K, windowLUT,
        windowHatInvLUT, FP, BP, tmpVec, B)
end

@inline dim(::NFFTPlan{D,DIM}) where {D,DIM} = DIM


##############
# constructors
##############
function NFFTPlan(x::Matrix{T}, N::NTuple{D,Int}, m = 4, sigma = 2.0,
                window=:kaiser_bessel, K=2000;
                precompute::PrecomputeFlags=LUT, sortNodes=false, kwargs...) where {D,T}

    if D != size(x,1)
        throw(ArgumentError("Nodes x have dimension $(size(x,1)) != $D"))
    end

    if any(isodd.(N))
      throw(ArgumentError("N = $N needs to consist of even integers!"))
    end

    n = ntuple(d->(ceil(Int,sigma*N[d])÷2)*2, D) # ensure that n is an even integer
    sigma = n[1] / N[1]

    tmpVec = Array{Complex{T},D}(undef, n)

    M = size(x, 2)

    FP = plan_fft!(tmpVec; kwargs...)
    BP = plan_bfft!(tmpVec; kwargs...)

    # Sort nodes in lexicographic way
    if sortNodes
        x .= sortslices(x,dims=2)
    end

    windowLUT, windowHatInvLUT, B = calcLookUpTable(x, N, n, m, sigma, window, K; precompute=precompute)

    NFFTPlan{D,0,T}(N, M, x, m, sigma, n, K, windowLUT, windowHatInvLUT, FP, BP, tmpVec, B)
end

function NFFTPlan!(p::AbstractNFFTPlan{D,DIM,T}, x::Matrix{T}, window = :kaiser_bessel; sortNodes=false) where {D,DIM,T}

    if isempty(p.B)
        precompute = LUT
    else
        precompute = FULL
    end

    # Sort nodes in lexicographic way
    if sortNodes
        x .= sortslices(x,dims=2)
      end

    windowLUT, windowHatInvLUT, B = calcLookUpTable(x, p.N, p.n, p.m, p.sigma, window, p.K; precompute=precompute)

    p.M = size(x, 2)
    p.windowLUT = windowLUT
    p.windowHatInvLUT = windowHatInvLUT
    p.B = B
    p.x = x

    return p
end

# Directional NFFT
"""
        NFFTPlan(x, d, N, ...) -> plan

Compute *directional* NFFT plan:
A 1D plan that is applied along dimension `d` of a `D` dimensional array of size `N` with sampling locations `x` (a vector).

It takes as optional keywords all the keywords supported by `plan_fft` function (like
`flags` and `timelimit`).  See documentation of `plan_fft` for reference.
"""
function NFFTPlan(x::AbstractVector{T}, dim::Integer, N::NTuple{D,Int64}, m=4,
                       sigma=2.0, window=:kaiser_bessel, K=2000; kwargs...) where {D,T}

    if any(isodd.(N))
      throw(ArgumentError("N = $N needs to consist of even integers!"))
    end

    n = ntuple(d->(ceil(Int,sigma*N[d])÷2)*2, D) # ensure that n is an even integer
    sigma = n[1] / N[1]

    sz = [N...]
    sz[dim] = n[dim]
    tmpVec = Array{Complex{T}}(undef, sz...)

    M = length(x)

    FP = plan_fft!(tmpVec, dim; kwargs...)
    BP = plan_bfft!(tmpVec, dim; kwargs...)

    windowLUT, windowHatInvLUT, B = calcLookUpTable(x, (N[dim],), (n[dim],), m, sigma, window, K, precompute=LUT)

    NFFTPlan{D,dim,T}(N, M, reshape(x, 1, M), m, sigma, n, K, windowLUT,
        windowHatInvLUT, FP, BP, tmpVec, B)
end

function Base.show(io::IO, p::NFFTPlan{D,0}) where {D}
    print(io, "NFFTPlan with ", p.M, " sampling points for ", p.N, " array")
end

function Base.show(io::IO, p::NFFTPlan{D,DIM}) where {D,DIM}
    print(io, "NFFTPlan with ", p.M, " sampling points for ", p.N, " array along dimension ", DIM)
end

size(p::NFFTPlan) = p.N
numFourierSamples(p::NFFTPlan) = p.M


################
# nfft functions
################
"""
        nfft!(p, f, fHat) -> fHat

Calculate the NFFT of `f` with plan `p` and store the result in `fHat`.

Both `f` and `fHat` must be complex arrays.
"""
function nfft!(p::NFFTPlan{D,DIM,T}, f::AbstractArray, fHat::StridedArray;
               verbose=false, timing::Union{Nothing,TimingStats} = nothing) where {D,DIM,T}
    consistencyCheck(p, f, fHat)

    fill!(p.tmpVec, zero(Complex{T}))
    t1 = @elapsed @inbounds apodization!(p, f, p.tmpVec)
    if nprocs() == 1
        t2 = @elapsed p.forwardFFT * p.tmpVec # fft!(p.tmpVec) or fft!(p.tmpVec, dim)
    else
        t2 = @elapsed dim(p) == 0 ? fft!(p.tmpVec) : fft!(p.tmpVec, dim(p))
    end
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
function nfft_adjoint!(p::NFFTPlan, fHat::AbstractArray, f::StridedArray;
                       verbose=false, timing::Union{Nothing,TimingStats} = nothing)
    consistencyCheck(p, f, fHat)

    t1 = @elapsed @inbounds convolve_adjoint!(p, fHat, p.tmpVec)
    if nprocs() == 1
        t2 = @elapsed p.backwardFFT * p.tmpVec # bfft!(p.tmpVec) or bfft!(p.tmpVec, dim)
    else
        t2 = @elapsed dim(p) == 0 ? bfft!(p.tmpVec) : bfft!(p.tmpVec, dim(p))
    end
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
