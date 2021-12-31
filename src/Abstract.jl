abstract type AbstractNFFTPlan{T,D,DIM} end

##################
# define interface
##################
# in-place NFFT
@mustimplement nfft!(p::AbstractNFFTPlan{D,DIM,T}, f::AbstractArray, fHat::AbstractArray) where {D,DIM,T}
# in-place adjoint NFFT
@mustimplement nfft_adjoint!(p::AbstractNFFTPlan{D,DIM,T}, fHat::AbstractArray, f::AbstractArray) where {D,DIM,T}
# size of the input array for the NFFT
@mustimplement size(p::AbstractNFFTPlan{D,DIM,T}) where {D,DIM,T}
# size of the output array for the NFFT
@mustimplement numFourierSamples(p::AbstractNFFTPlan{D,DIM,T}) where {D,DIM,T}

#################
# derived methods
#################
"""
        nfft(p, f) -> fHat

For a **non**-directional `D` dimensional plan `p` this calculates the NFFT of a `D` dimensional array `f` of size `N`.
`fHat` is a vector of length `M`.
(`M` and `N` are defined in the plan creation)

For a **directional** `D` dimensional plan `p` both `f` and `fHat` are `D`
dimensional arrays, and the dimension specified in the plan creation is
affected.
"""
function nfft(p::AbstractNFFTPlan{D,0,T}, f::AbstractArray{U,D}, args...; kargs...) where {D,T,U}
    fHat = similar(f,Complex{T}, p.M)
    nfft!(p, f, fHat, args...; kargs...)
    return fHat
end

function nfft(p::AbstractNFFTPlan{D,DIM,T}, f::AbstractArray{U,D}, args...; kargs...) where {D,DIM,T,U}
    sz = [p.N...]
    sz[DIM] = p.M
    fHat = similar(f, Complex{T}, Tuple(sz))
    nfft!(p, f, fHat, args...; kargs...)
    return fHat
end

"""
        nfft_adjoint(p, fHat) -> f

For a **non**-directional `D` dimensional plan `p` this calculates the adjoint NFFT of a length `M` vector `fHat`
`f` is a `D` dimensional array of size `N`.
(`M` and `N` are defined in the plan creation)

For a **directional** `D` dimensional plan `p` both `f` and `fHat` are `D`
dimensional arrays, and the dimension specified in the plan creation is
affected.
"""
function nfft_adjoint(p::AbstractNFFTPlan{D,DIM,T}, fHat::AbstractArray{U}, args...; kargs...) where {D,DIM,T,U}
    f = similar(fHat, Complex{T}, p.N)
    nfft_adjoint!(p, fHat, f, args...; kargs...)
    return f
end

