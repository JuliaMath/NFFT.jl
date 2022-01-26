
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

plan_nfft(Q::Type, x::AbstractVector, N::NTuple{D,Int}, rest...; kwargs...) where {D}  =
    plan_nfft(Q, collect(reshape(x,1,length(x))), N, rest...; kwargs...)

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
function nfft(x, f::AbstractArray{T,D}, rest...;  kwargs...) where {T,D}
  p = plan_nfft(x, size(f), rest...; kwargs... )
  return nfft(p, f)
end

"""
nfft_adjoint(x, N, fHat::AbstractArray{T,D}, rest...; kwargs...)

calculates the adjoint NFFT of the vector `fHat` for the nodes contained in the matrix `x`.
The output is an array of size `N`
"""
function nfft_adjoint(x, N, fHat::AbstractVector{T}, rest...;  kwargs...) where T
  p = plan_nfft(x, N, rest...;  kwargs...)
  return nfft_adjoint(p, fHat)
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
function nfft(p::AbstractNFFTPlan{T,D,R}, f::AbstractArray{U,D}, args...; kargs...) where {T,D,R,U}
    fHat = similar(f, Complex{T}, size_out(p))
    nfft!(p, f, fHat, args...; kargs...)
    return fHat
end

function nfft(p::AbstractNNFFTPlan{T,D,R}, f::AbstractArray{U,D}, args...; kargs...) where {T,D,R,U}
  fHat = similar(f, Complex{T}, size_out(p))
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
function nfft_adjoint(p::AbstractNFFTPlan{T,D,R}, fHat::AbstractArray{U}, args...; kargs...) where {T,D,R,U}
    f = similar(fHat, Complex{T}, size_in(p))
    nfft_adjoint!(p, fHat, f, args...; kargs...)
    return f
end

function nfft_adjoint(p::AbstractNNFFTPlan{T}, fHat::AbstractArray, args...; kargs...) where {T}
  f = similar(fHat, Complex{T}, size_in(p))
  nfft_adjoint!(p, fHat, f, args...; kargs...)
  return f
end

##########################
# Linear Algebra wrappers
##########################

Base.eltype(p::AbstractNFFTPlan{T}) where T = Complex{T}
Base.size(p::AbstractNFFTPlan) = (prod(size_out(p)), prod(size_in(p)))
Base.size(p::Adjoint{Complex{T}, U}) where {T, U<:AbstractNFFTPlan{T}} = 
  (prod(size_in(p.parent)), prod(size_out(p.parent)))

LinearAlgebra.adjoint(p::AbstractNFFTPlan{T,D,R}) where {T,D,R} = 
Adjoint{Complex{T}, typeof(p)}(p)

LinearAlgebra.mul!(C::AbstractArray, A::AbstractNFFTPlan, B::AbstractArray; kargs...) =
   nfft!(A, B, C; kargs...)

LinearAlgebra.mul!(C::AbstractArray, A::Adjoint{Complex{T},U}, B::AbstractArray; kargs...) where {T, U<:AbstractNFFTPlan{T}} =
   nfft_adjoint!(A.parent, B, C; kargs...)

function Base.:*(A::AbstractNFFTPlan, B::AbstractArray; kargs...)
   nfft(A, B; kargs...)
end

function Base.:*(A::Adjoint{Complex{T},<:AbstractNFFTPlan{T}}, B::AbstractVector{Complex{T}}; kargs...) where {T}
  nfft_adjoint(A.parent, B; kargs...)
end

function Base.:*(A::Adjoint{Complex{T},<:AbstractNFFTPlan{T}}, B::AbstractArray{Complex{T},D}; kargs...) where {T, D}
   nfft_adjoint(A.parent, B; kargs...)
end