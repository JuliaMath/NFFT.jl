
##########################
# plan_nfft constructors
##########################

# The following automatically call the plan_nfft version for type Array

plan_nfft(x::AbstractArray, N::Union{Integer,NTuple{D,Int}}, args...; kargs...) where {D} =
    plan_nfft(Array, x, N, args...; kargs...)

plan_nfft(x::AbstractArray, y::AbstractArray, args...; kargs...) where {D} =
    plan_nfft(Array, x, y, args...; kargs...)

# The follow convert 1D parameters into the format required by the NFFT plan

plan_nfft(Q::Type, x::AbstractVector, N::Integer, rest...; kwargs...) where {D}  =
    plan_nfft(Q, collect(reshape(x,1,length(x))), (N,), rest...; kwargs...)

plan_nfft(Q::Type, x::AbstractVector, N::NTuple{D,Int}, rest...; kwargs...) where {D} =
    plan_nfft(Q, collect(reshape(x,1,length(x))), N, rest...; kwargs...) 

plan_nfft(Q::Type, x::AbstractMatrix, N::NTuple{D,Int}, rest...; kwargs...) where {D}  =
    plan_nfft(Q, collect(x), N, rest...; kwargs...)

plan_nfft(Q::Type, x::AbstractVector, y::AbstractVector, rest...; kwargs...) where {D}  =
    plan_nfft(Q, collect(reshape(x,1,length(x))), collect(reshape(y,1,length(x))), rest...; kwargs...)


##########################
# Allocating nfft functions
##########################

"""
nfft(x, f::AbstractArray{T,D}, rest...; kwargs...)

calculates the NFFT of the array `f` for the nodes contained in the matrix `x`
The output is a vector of length M=`size(nodes,2)`
"""
function nfft(x, f::AbstractArray{T,D}; kargs...) where {T,D}
  p = plan_nfft(x, size(f); kargs... )
  return p * f
end

"""
nfft_adjoint(x, N, fHat::AbstractArray{T,D}, rest...; kwargs...)

calculates the adjoint NFFT of the vector `fHat` for the nodes contained in the matrix `x`.
The output is an array of size `N`
"""
function nfft_adjoint(x, N, fHat::AbstractVector{T};  kargs...) where T
  p = plan_nfft(x, N;  kargs...)
  return adjoint(p) * fHat
end


"""
        *(p, f) -> fHat

For a **non**-directional `D` dimensional plan `p` this calculates the NFFT of a `D` dimensional array `f` of size `N`.
`fHat` is a vector of length `M`.
(`M` and `N` are defined in the plan creation)

For a **directional** `D` dimensional plan `p` both `f` and `fHat` are `D`
dimensional arrays, and the dimension specified in the plan creation is
affected.
"""
function Base.:*(p::AnyNFFTPlan{T}, f::AbstractArray{Complex{U},D}; kargs...) where {T,U,D}
  fHat = similar(f, Complex{T}, size_out(p))
  mul!(fHat, p, f; kargs...)
  return fHat
end

"""
        *(p::Adjoint{T,<:AnyNFFTPlan{T}}, fHat) -> f

For a **non**-directional `D` dimensional plan `p` this calculates the adjoint NFFT of a length `M` vector `fHat`
`f` is a `D` dimensional array of size `N`.
(`M` and `N` are defined in the plan creation)

For a **directional** `D` dimensional plan `p` both `f` and `fHat` are `D`
dimensional arrays, and the dimension specified in the plan creation is
affected.
"""

function Base.:*(p::Adjoint{Complex{T},<:AnyNFFTPlan{T}}, fHat::AbstractArray{Complex{U},D}; kargs...) where {T,U,D}
  f = similar(fHat, Complex{T}, size_out(p))
  mul!(f, p, fHat; kargs...)
  return f
end

# The following two methods are redundant but need to be defined because of a method ambiguity with Julia Base
function Base.:*(p::Adjoint{Complex{T},<:AnyNFFTPlan{T}}, fHat::AbstractVector{Complex{U}}; kargs...) where {T,U}
  f = similar(fHat, Complex{T}, size_out(p))
  mul!(f, p, fHat; kargs...)
  return f
end
function Base.:*(p::Adjoint{Complex{T},<:AnyNFFTPlan{T}}, fHat::AbstractArray{Complex{U},2}; kargs...) where {T,U}
  f = similar(fHat, Complex{T}, size_out(p))
  mul!(f, p, fHat; kargs...)
  return f
end


