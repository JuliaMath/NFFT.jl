module NFFT

using Base.Cartesian
using FFTW
using Distributed

export NFFTPlan, nfft, nfft_adjoint, ndft, ndft_adjoint

include("windowFunctions.jl")

#=
Some internal documentation (especially for people familiar with the nfft)

- Currently the window cannot be changed and defaults to the kaiser-bessel
  window. This is done for simplicity and due to the fact that the
  kaiser-bessel window usually outperforms any other window

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
mutable struct NFFTPlan{D,DIM,T}
    N::NTuple{D,Int64}
    M::Int64
    x::Matrix{T}
    m::Int64
    sigma::T
    n::NTuple{D,Int64}
    K::Int64
    windowLUT::Vector{Vector{T}}
    windowHatInvLUT::Vector{Vector{T}}
    forwardFFT::FFTW.cFFTWPlan{Complex{Float64},-1,true,D}
    backwardFFT::FFTW.cFFTWPlan{Complex{Float64},1,true,D}
    tmpVec::Array{Complex{T},D}
end

@inline dim(::NFFTPlan{D,DIM}) where {D, DIM} = DIM

"""
        NFFTPlan(x, N, ...) -> plan

Compute `D` dimensional NFFT plan for sampling locations `x` (a vector or a `D`-by-`M` matrix) that can be applied on arrays of size `N` (a tuple of length `D`).

The optional arguments control the accuracy.

It takes as optional keywords all the keywords supported by `plan_fft` function (like
`flags` and `timelimit`).  See documentation of `plan_fft` for reference.
"""
NFFTPlan

function NFFTPlan(x::Matrix{T}, N::NTuple{D,Int}, m=4, sigma=2.0,
                       window=:kaiser_bessel, K=2000; kwargs...) where {D,T}
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
    Z = round(Int,3*K/2)
    for d=1:D
        windowLUT[d] = zeros(T, Z)
        for l=1:Z
            y = ((l-1) / (K-1)) * m/n[d]
            windowLUT[d][l] = win(y, n[d], m, sigma)
        end
    end

    windowHatInvLUT = Vector{Vector{T}}(undef,D)
    for d=1:D
        windowHatInvLUT[d] = zeros(T, N[d])
        for k=1:N[d]
            windowHatInvLUT[d][k] = 1. / win_hat(k-1-N[d]/2, n[d], m, sigma)
        end
    end

    NFFTPlan{D,0,T}(N, M, x, m, sigma, n, K, windowLUT, windowHatInvLUT, FP, BP, tmpVec )
end

NFFTPlan(x::AbstractMatrix{T}, N::NTuple{D,Int}, rest...; kwargs...) where {D,T} =
    NFFTPlan(collect(x), N, rest...; kwargs...)

NFFTPlan(x::AbstractVector, N::Integer, rest...; kwargs...) =
    NFFTPlan(reshape(x,1,length(x)), (N,), rest...; kwargs...)

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

    NFFTPlan{D,dim,T}(N, M, reshape(x,1,M), m, sigma, n, K, windowLUT, windowHatInvLUT, FP, BP, tmpVec)
end

function NFFTPlan(x::Matrix{T}, dim::Integer, N::NTuple{D,Int}, m=4, sigma=2.0,
                       window=:kaiser_bessel, K=2000; kwargs...) where {D,T}
    if size(x,1) != 1 && size(x,2) != 1
        throw(DimensionMismatch())
    end

    NFFTPlan(vec(x), dim, N, m, sigma, window, K; kwargs...)
end


function Base.show(io::IO, p::NFFTPlan{D,0}) where D
    print(io, "NFFTPlan with ", p.M, " sampling points for ", p.N, " array")
end

function Base.show(io::IO, p::NFFTPlan{D,DIM}) where {D,DIM}
    print(io, "NFFTPlan with ", p.M, " sampling points for ", p.N, " array along dimension ", DIM)
end


@generated function consistencyCheck(p::NFFTPlan{D,DIM}, f::AbstractArray{T,D},
                                     fHat::AbstractArray{T}) where {D,DIM,T}
    quote
        if $DIM == 0
            fHat_test = (p.M == length(fHat))
        elseif $DIM > 0
            fHat_test = @nall $D d -> ( d == $DIM ? size(fHat,d) == p.M : size(fHat,d) == p.N[d] )
        end

        if p.N != size(f) || !fHat_test
            throw(DimensionMismatch("Data is not consistent with NFFTPlan"))
        end
    end
end


### nfft functions ###

"""
        nfft!(p, f, fHat) -> fHat

Calculate the NFFT of `f` with plan `p` and store the result in `fHat`.

Both `f` and `fHat` must be complex arrays.
"""
function nfft!(p::NFFTPlan, f::AbstractArray{T}, fHat::StridedArray{T}) where T
    consistencyCheck(p, f, fHat)

    fill!(p.tmpVec, zero(T))
    @inbounds apodization!(p, f, p.tmpVec)
    if nprocs() == 1
        p.forwardFFT * p.tmpVec # fft!(p.tmpVec) or fft!(p.tmpVec, dim)
    else
        dim(p) == 0 ? fft!(p.tmpVec) : fft!(p.tmpVec, dim(p))
    end
    @inbounds convolve!(p, p.tmpVec, fHat)
    return fHat
end

"""
        nfft(p, f) -> fHat

For a **non**-directional `D` dimensional plan `p` this calculates the NFFT of a `D` dimensional array `f` of size `N`.
`fHat` is a vector of length `M`.
(`M` and `N` are defined in the plan creation)

For a **directional** `D` dimensional plan `p` both `f` and `fHat` are `D`
dimensional arrays, and the dimension specified in the plan creation is
affected.
"""
function nfft(p::NFFTPlan{D,0}, f::AbstractArray{T,D}) where {D,T}
    fHat = zeros(T, p.M)
    nfft!(p, f, fHat)
    return fHat
end

function nfft(x, f::AbstractArray{T,D}, rest...; kwargs...) where {D,T}
    p = NFFTPlan(x, size(f), rest...; kwargs...)
    return nfft(p, f)
end

function nfft(p::NFFTPlan{D,DIM}, f::AbstractArray{T,D}) where {D,DIM,T}
    sz = [p.N...]
    sz[DIM] = p.M
    fHat = Array{T}(undef,sz...)
    nfft!(p, f, fHat)
    return fHat
end


"""
        nfft_adjoint!(p, fHat, f) -> f

Calculate the adjoint NFFT of `fHat` and store the result in `f`.

Both `f` and `fHat` must be complex arrays.
"""
function nfft_adjoint!(p::NFFTPlan, fHat::AbstractArray, f::StridedArray)
    consistencyCheck(p, f, fHat)

    @inbounds convolve_adjoint!(p, fHat, p.tmpVec)
    if nprocs() == 1
        p.backwardFFT * p.tmpVec # bfft!(p.tmpVec) or bfft!(p.tmpVec, dim)
    else
        dim(p) == 0 ? bfft!(p.tmpVec) : bfft!(p.tmpVec, dim(p))
    end
    @inbounds apodization_adjoint!(p, p.tmpVec, f)
    return f
end

"""
        nfft_adjoint(p, f) -> fHat

For a **non**-directional `D` dimensional plan `p` this calculates the adjoint NFFT of a length `M` vector `fHat`
`f` is a `D` dimensional array of size `N`.
(`M` and `N` are defined in the plan creation)

For a **directional** `D` dimensional plan `p` both `f` and `fHat` are `D`
dimensional arrays, and the dimension specified in the plan creation is
affected.
"""
function nfft_adjoint(p::NFFTPlan{D,DIM}, fHat::AbstractArray{T}) where {D,DIM,T}
    f = Array{T}(undef,p.N)
    nfft_adjoint!(p, fHat, f)
    return f
end

function nfft_adjoint(x, N, fHat::AbstractVector{T}, rest...; kwargs...) where T
    p = NFFTPlan(x, N, rest...; kwargs...)
    return nfft_adjoint(p, fHat)
end


### ndft functions ###

function ndft(plan::NFFTPlan{D}, f::AbstractArray{T,D}) where {D,T}
    plan.N == size(f) || throw(DimensionMismatch("Data is not consistent with NFFTPlan"))

    g = zeros(T, plan.M)

    for l=1:prod(plan.N)
        idx = CartesianIndices(plan.N)[l]

        for k=1:plan.M
            arg = zero(T)
            for d=1:D
                arg += plan.x[d,k] * ( idx[d] - 1 - plan.N[d] / 2 )
            end
            g[k] += f[l] * cis(-2*pi*arg)
        end
    end

    return g
end

function ndft_adjoint(plan::NFFTPlan{D}, fHat::AbstractArray{T,1}) where {D,T}
    plan.M == length(fHat) || throw(DimensionMismatch("Data is not consistent with NFFTPlan"))

    g = zeros(T, plan.N)

    for l=1:prod(plan.N)
        idx = CartesianIndices(plan.N)[l]

        for k=1:plan.M
            arg = zero(T)
            for d=1:D
                arg += plan.x[d,k] * ( idx[d] - 1 - plan.N[d] / 2 )
            end
            g[l] += fHat[k] * cis(2*pi*arg)
        end
    end

    return g
end



### convolve! ###

function convolve!(p::NFFTPlan{1,0}, g::AbstractVector{T}, fHat::StridedVector{T}) where T
    fill!(fHat, zero(T))
    scale = 1.0 / p.m * (p.K-1)
    n = p.n[1]

    for k=1:p.M # loop over nonequispaced nodes
        c = floor(Int, p.x[k]*n)
        for l=(c-p.m):(c+p.m) # loop over nonzero elements
            gidx = rem(l+n, n) + 1
            idx = abs( (p.x[k]*n - l)*scale ) + 1
            idxL = floor(Int, idx)

            fHat[k] += g[gidx] * (p.windowLUT[1][idxL] + ( idx-idxL ) * (p.windowLUT[1][idxL+1] - p.windowLUT[1][idxL]))
        end
    end
end

function convolve!(p::NFFTPlan{D,0}, g::AbstractArray{T,D}, fHat::StridedVector{T}) where {D,T}
    scale = 1.0 / p.m * (p.K-1)

    Threads.@threads for k in 1:p.M
        fHat[k] = _convolve(p, g, scale, k)
    end
end


@generated function _convolve(p::NFFTPlan{D,0}, g::AbstractArray{T,D}, scale, k) where {D,T}
    quote
        @nexprs $D d -> xscale_d = p.x[d,k] * p.n[d]
        @nexprs $D d -> c_d = floor(Int, xscale_d)

        fHat = zero(T)

        @nloops $D l d -> (c_d-p.m):(c_d+p.m) d->begin
            # preexpr
            gidx_d = rem(l_d+p.n[d], p.n[d]) + 1
            idx = abs( (xscale_d - l_d)*scale ) + 1
            idxL = floor(idx)
            idxInt = Int(idxL)
            tmpWin_d = p.windowLUT[d][idxInt] + ( idx-idxL ) * (p.windowLUT[d][idxInt+1] - p.windowLUT[d][idxInt])
        end begin
            # bodyexpr
            v = @nref $D g gidx
            @nexprs $D d -> v *= tmpWin_d
            fHat += v
        end

        return fHat
    end
end

@generated function convolve!(p::NFFTPlan{D,DIM}, g::AbstractArray{T,D}, fHat::StridedArray{T,D}) where {D,DIM,T}
    quote
        fill!(fHat, zero(T))
        scale = 1.0 / p.m * (p.K-1)

        for k in 1:p.M
            xscale = p.x[k] * p.n[$DIM]
            c = floor(Int, xscale)
            @nloops $D l d->begin
                # rangeexpr
                if d == $DIM
                    (c-p.m):(c+p.m)
                else
                    1:size(g,d)
                end
            end d->begin
                # preexpr
                if d == $DIM
                    gidx_d = rem(l_d+p.n[d], p.n[d]) + 1
                    idx = abs( (xscale - l_d)*scale ) + 1
                    idxL = floor(idx)
                    idxInt = Int(idxL)
                    tmpWin = p.windowLUT[1][idxInt] + ( idx-idxL ) * (p.windowLUT[1][idxInt+1] - p.windowLUT[1][idxInt])
                    fidx_d = k
                else
                    gidx_d = l_d
                    fidx_d = l_d
                end
            end begin
                # bodyexpr
                (@nref $D fHat fidx) += (@nref $D g gidx) * tmpWin
            end
        end
    end
end


### convolve_adjoint! ###

function convolve_adjoint!(p::NFFTPlan{1,0}, fHat::AbstractVector{T}, g::StridedVector{T}) where T
    fill!(g, zero(T))
    scale = 1.0 / p.m * (p.K-1)
    n = p.n[1]

    for k=1:p.M # loop over nonequispaced nodes
        c = round(Int,p.x[k]*n)
        for l=(c-p.m):(c+p.m) # loop over nonzero elements
            gidx = rem(l+n, n) + 1
            idx = abs( (p.x[k]*n - l)*scale ) + 1
            idxL = round(Int, idx)

            g[gidx] += fHat[k] * (p.windowLUT[1][idxL] + ( idx-idxL ) * (p.windowLUT[1][idxL+1] - p.windowLUT[1][idxL]))
        end
    end
end

@generated function convolve_adjoint!(p::NFFTPlan{D,0}, fHat::AbstractVector{T}, g::StridedArray{T,D}) where {D,T}
    quote
        fill!(g, zero(T))
        scale = 1.0 / p.m * (p.K-1)

        for k in 1:p.M
            @nexprs $D d -> xscale_d = p.x[d,k] * p.n[d]
            @nexprs $D d -> c_d = floor(Int, xscale_d)

            @nloops $D l d -> (c_d-p.m):(c_d+p.m) d->begin
                # preexpr
                gidx_d = rem(l_d+p.n[d], p.n[d]) + 1
                idx = abs( (xscale_d - l_d)*scale ) + 1
                idxL = floor(idx)
                idxInt = Int(idxL)
                tmpWin_d = p.windowLUT[d][idxInt] + ( idx-idxL ) * (p.windowLUT[d][idxInt+1] - p.windowLUT[d][idxInt])
            end begin
                # bodyexpr
                v = fHat[k]
                @nexprs $D d -> v *= tmpWin_d
                (@nref $D g gidx) += v
            end
        end
    end
end

@generated function convolve_adjoint!(p::NFFTPlan{D,DIM}, fHat::AbstractArray{T,D}, g::StridedArray{T,D}) where {D,DIM,T}
    quote
        fill!(g, zero(T))
        scale = 1.0 / p.m * (p.K-1)

        for k in 1:p.M
            xscale = p.x[k] * p.n[$DIM]
            c = floor(Int, xscale)
            @nloops $D l d->begin
                # rangeexpr
                if d == $DIM
                    (c-p.m):(c+p.m)
                else
                    1:size(g,d)
                end
            end d->begin
                # preexpr
                if d == $DIM
                    gidx_d = rem(l_d+p.n[d], p.n[d]) + 1
                    idx = abs( (xscale - l_d)*scale ) + 1
                    idxL = floor(idx)
                    idxInt = Int(idxL)
                    tmpWin = p.windowLUT[1][idxInt] + ( idx-idxL ) * (p.windowLUT[1][idxInt+1] - p.windowLUT[1][idxInt])
                    fidx_d = k
                else
                    gidx_d = l_d
                    fidx_d = l_d
                end
            end begin
                # bodyexpr
                (@nref $D g gidx) += (@nref $D fHat fidx) * tmpWin
            end
        end
    end
end


### apodization! ###

function apodization!(p::NFFTPlan{1,0}, f::AbstractVector{T}, g::StridedVector{T}) where T
    n = p.n[1]
    N = p.N[1]
    offset = round( Int, n - N / 2 ) - 1
    for l=1:N
        g[((l+offset)% n) + 1] = f[l] * p.windowHatInvLUT[1][l]
    end
end

@generated function apodization!(p::NFFTPlan{D,0}, f::AbstractArray{T,D}, g::StridedArray{T,D}) where {D,T}
    quote
        @nexprs $D d -> offset_d = round(Int, p.n[d] - p.N[d]/2) - 1

        @nloops $D l f d->(gidx_d = rem(l_d+offset_d, p.n[d]) + 1) begin
            v = @nref $D f l
            @nexprs $D d -> v *= p.windowHatInvLUT[d][l_d]
            (@nref $D g gidx) = v
        end
    end
end

@generated function apodization!(p::NFFTPlan{D,DIM}, f::AbstractArray{T,D}, g::StridedArray{T,D}) where {D,DIM,T}
    quote
        offset = round(Int, p.n[$DIM] - p.N[$DIM]/2) - 1

        @nloops $D l f d->begin
            # preexpr
            if d == $DIM
                gidx_d = rem(l_d+offset, p.n[d]) + 1
                winidx = l_d
            else
                gidx_d = l_d
            end
        end begin
            # bodyexpr
            (@nref $D g gidx) = (@nref $D f l) * p.windowHatInvLUT[1][winidx]
        end
    end
end


### apodization_adjoint! ###

function apodization_adjoint!(p::NFFTPlan{1,0}, g::AbstractVector{T}, f::StridedVector{T}) where T
    n = p.n[1]
    N = p.N[1]
    offset = round( Int, n - N / 2 ) - 1
    for l=1:N
        f[l] = g[((l+offset)% n) + 1] * p.windowHatInvLUT[1][l]
    end
end

@generated function apodization_adjoint!(p::NFFTPlan{D,0}, g::AbstractArray{T,D}, f::StridedArray{T,D}) where {D,T}
    quote
        @nexprs $D d -> offset_d = round(Int, p.n[d] - p.N[d]/2) - 1

        @nloops $D l f begin
            v = @nref $D g d -> rem(l_d+offset_d, p.n[d]) + 1
            @nexprs $D d -> v *= p.windowHatInvLUT[d][l_d]
            (@nref $D f l) = v
        end
    end
end

@generated function apodization_adjoint!(p::NFFTPlan{D,DIM}, g::AbstractArray{T,D}, f::StridedArray{T,D}) where {D,DIM,T}
    quote
        offset = round(Int, p.n[$DIM] - p.N[$DIM]/2) - 1

        @nloops $D l f d->begin
            # preexpr
            if d == $DIM
                gidx_d = rem(l_d+offset, p.n[d]) + 1
                winidx = l_d
            else
                gidx_d = l_d
            end
        end begin
            # bodyexpr
            (@nref $D f l) = (@nref $D g gidx) * p.windowHatInvLUT[1][winidx]
        end
    end
end


include("samplingDensity.jl")

end
