
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
    
    FP = plan_fft!(tmpVec; flags=p.forwardFFT.flags)
    BP = plan_bfft!(tmpVec; flags=p.backwardFFT.flags)
   
    return  NFFTPlan{T,D,0}(p.N, p.M, p.x, p.m, p.sigma, p.n, p.K, p.windowLUT,
                    p.windowHatInvLUT, FP, BP , tmpVec, p.B)
end
    
function Base.copy(p::NFFTPlan{D,DIM,T}) where {D,DIM,T}
    tmpVec = similar(p.tmpVec)
    
    FP = plan_fft!(tmpVec, DIM; flags=p.forwardFFT.flags)
    BP = plan_bfft!(tmpVec, DIM; flags=p.backwardFFT.flags)
    
    return  NFFTPlan{T,D,DIM}(p.N, p.M, p.x, p.m, p.sigma, p.n, p.K, p.windowLUT,
                    p.windowHatInvLUT, FP, BP , tmpVec, p.B)
end
    
@inline dim(::NFFTPlan{D,DIM}) where {D, DIM} = DIM
    

##############
# constructors
##############
function NFFTPlan(x::Matrix{T}, N::NTuple{D,Int}, m=4, sigma=2.0,
                window=:kaiser_bessel, K=2000;
                precompute::PrecomputeFlags=LUT, kwargs...) where {D,T}

    if D != size(x,1)
        throw(ArgumentError())
    end

    n = ntuple(d->round(Int,sigma*N[d]), D)

    tmpVec = zeros(Complex{T}, n)

    M = size(x,2)

    FP = plan_fft!(tmpVec; kwargs...)
    BP = plan_bfft!(tmpVec; kwargs...)

    # Create lookup table
    win, win_hat = getWindow(window)

    windowLUT = Vector{Vector{T}}(undef,D)
    windowHatInvLUT = Vector{Vector{T}}(undef,D)
    for d=1:D
        windowHatInvLUT[d] = zeros(T, N[d])
        for k=1:N[d]
            windowHatInvLUT[d][k] = 1. / win_hat(k-1-N[d]/2, n[d], m, sigma)
        end
    end

    if precompute == LUT
        Z = round(Int,3*K/2)
        for d=1:D
            windowLUT[d] = zeros(T, Z)
            for l=1:Z
                y = ((l-1) / (K-1)) * m/n[d]
                windowLUT[d][l] = win(y, n[d], m, sigma)
            end
        end

        B = sparse([],[],Float64[])
    else
        U1 = ntuple(d-> (d==1) ? 1 : (2*m+1)^(d-1), D)
        U2 = ntuple(d-> (d==1) ? 1 : prod(n[1:(d-1)]), D)
        B = precomputeB(win, x, n, m, M, sigma, T, U1, U2)
    end

    NFFTPlan{D,0,T}(N, M, x, m, sigma, n, K, windowLUT, windowHatInvLUT, FP, BP, tmpVec, B )
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
    n = ntuple(d->round(Int, sigma*N[d]), D)

    sz = [N...]
    sz[dim] = n[dim]
    tmpVec = Array{Complex{T}}(undef,sz...)

    M = length(x)

    FP = plan_fft!(tmpVec, dim; kwargs...)
    BP = plan_bfft!(tmpVec, dim; kwargs...)

    # Create lookup table
    win, win_hat = getWindow(window)

    windowLUT = Vector{Vector{T}}(undef,1)
    Z = round(Int, 3*K/2)
    windowLUT[1] = zeros(T, Z)
    for l = 1:Z
        y = ((l-1) / (K-1)) * m/n[dim]
        windowLUT[1][l] = win(y, n[dim], m, sigma)
    end

    windowHatInvLUT = Vector{Vector{T}}(undef,1)
    windowHatInvLUT[1] = zeros(T, N[dim])
    for k = 1:N[dim]
        windowHatInvLUT[1][k] = 1. / win_hat(k-1-N[dim]/2, n[dim], m, sigma)
    end

    B = sparse([],[],Float64[])

    NFFTPlan{D,dim,T}(N, M, reshape(x,1,M), m, sigma, n, K, windowLUT,
                      windowHatInvLUT, FP, BP, tmpVec, B)
end

function Base.show(io::IO, p::NFFTPlan{D,0}) where D
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
function nfft!(p::NFFTPlan{D,DIM,T}, f::AbstractArray, fHat::StridedArray, verbose=false) where {D,DIM,T}
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
    return fHat
end

"""
        nfft_adjoint!(p, fHat, f) -> f

Calculate the adjoint NFFT of `fHat` and store the result in `f`.

Both `f` and `fHat` must be complex arrays.
"""
function nfft_adjoint!(p::NFFTPlan, fHat::AbstractArray, f::StridedArray, verbose=false)
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
    return f
end
