"""
  AnyNFFTPlan{T,D,R}

Abstract type for either an NFFT plan or an NNFT plan.
* T is the element type (Float32/Float64)
* D is the number of dimensions of the input array.
* R is the number of dimensions of the output array.
"""
abstract type AnyNFFTPlan{T,D,R} end

"""
  AbstractNFFTPlan{T,D,R}

Abstract type for an NFFT plan.
* T is the element type (Float32/Float64)
* D is the number of dimensions of the input array.
* R is the number of dimensions of the output array.
"""
abstract type AbstractNFFTPlan{T,D,R} <: AnyNFFTPlan{T,D,R} end

"""
  AbstractNFCTPlan{T,D,R}

Abstract type for an NFCT plan.
* T is the element type (Float32/Float64)
* D is the number of dimensions of the input array.
* R is the number of dimensions of the output array.
"""
abstract type AbstractNFCTPlan{T,D,R} <: AnyNFFTPlan{T,D,R} end

"""
  AbstractNNFFTPlan{T,D,R}

Abstract type for an NNFFT plan.
* T is the element type (Float32/Float64)
* D is the number of dimensions of the input array.
* R is the number of dimensions of the output array.
"""
abstract type AbstractNNFFTPlan{T,D,R} <: AnyNFFTPlan{T,D,R} end

#####################
# Function needed to make AnyNFFTPlan an operator 
#####################

Base.eltype(p::AbstractNFFTPlan{T}) where T = Complex{T}
Base.eltype(p::AbstractNNFFTPlan{T}) where T = Complex{T}
Base.eltype(p::AbstractNFCTPlan{T}) where T = T

Base.size(p::AnyNFFTPlan) = (prod(size_out(p)), prod(size_in(p)))

Base.size(p::Adjoint{Complex{T}, U}) where {T, U<:AbstractNFFTPlan{T}} = 
  (prod(size_in(p.parent)), prod(size_out(p.parent)))

Base.size(p::Adjoint{Complex{T}, U}) where {T, U<:AbstractNNFFTPlan{T}} = 
  (prod(size_in(p.parent)), prod(size_out(p.parent)))

Base.size(p::Transpose{T, U}) where {T, U<:AbstractNFCTPlan{T}} = 
  (prod(size_in(p.parent)), prod(size_out(p.parent)))

LinearAlgebra.adjoint(p::AbstractNFFTPlan{T,D,R}) where {T,D,R} = 
Adjoint{Complex{T}, typeof(p)}(p)

LinearAlgebra.adjoint(p::AbstractNNFFTPlan{T,D,R}) where {T,D,R} = 
Adjoint{Complex{T}, typeof(p)}(p)

LinearAlgebra.transpose(p::AbstractNFCTPlan{T,D,R}) where {T,D,R} = 
Transpose{T, typeof(p)}(p)

size_in(p::Adjoint{Complex{T},<:AbstractNFFTPlan{T,D,R}}) where {T,D,R} = size_out(p.parent)
size_out(p::Adjoint{Complex{T},<:AbstractNFFTPlan{T,D,R}}) where {T,D,R} = size_in(p.parent)

size_in(p::Adjoint{Complex{T},<:AbstractNNFFTPlan{T,D,R}}) where {T,D,R} = size_out(p.parent)
size_out(p::Adjoint{Complex{T},<:AbstractNNFFTPlan{T,D,R}}) where {T,D,R} = size_in(p.parent)

size_in(p::Transpose{T,<:AbstractNFCTPlan{T,D,R}}) where {T,D,R} = size_out(p.parent)
size_out(p::Transpose{T,<:AbstractNFCTPlan{T,D,R}}) where {T,D,R} = size_in(p.parent)

#####################
# define interface
#####################



"""
    mul!(fHat, p, f) -> fHat

Inplace NFFT transforming the `D` dimensional array `f` to the `R` dimensional array `fHat`.
The transformation is applied along `D-R+1` dimensions specified in the plan `p`.
Both `f` and `fHat` must be complex arrays of element type `Complex{T}`.
"""
@mustimplement LinearAlgebra.mul!( fHat::AbstractArray, p::AbstractNFFTPlan{T}, f::AbstractArray) where {T}

"""
    mul!(fHat, p, f) -> fHat

Inplace NFCT transforming the `D` dimensional array `f` to the `R` dimensional array `fHat`.
The transformation is applied along `D-R+1` dimensions specified in the plan `p`.
Both `f` and `fHat` must be complex arrays of element type `Complex{T}`.
"""
@mustimplement LinearAlgebra.mul!( fHat::AbstractArray, p::AbstractNFCTPlan{T}, f::AbstractArray) where {T}

"""
    mul!(f, p, fHat) -> f

Inplace adjoint NFFT transforming the `R` dimensional array `fHat` to the `D` dimensional array `f`.
The transformation is applied along `D-R+1` dimensions specified in the plan `p`.
Both `f` and `fHat` must be complex arrays of element type `Complex{T}`.
"""
@mustimplement LinearAlgebra.mul!(f::AbstractArray, p::Adjoint{Complex{T},<:AbstractNFFTPlan{T}}, fHat::AbstractArray) where {T}

"""
    mul!(f, p, fHat) -> f

Inplace transposed NFCT transforming the `R` dimensional array `fHat` to the `D` dimensional array `f`.
The transformation is applied along `D-R+1` dimensions specified in the plan `p`.
Both `f` and `fHat` must be arrays of element type `T`.
"""
@mustimplement LinearAlgebra.mul!(f::AbstractArray, p::Transpose{T,<:AbstractNFCTPlan{T}}, fHat::AbstractArray) where {T}

"""
    size_in(p)

Size of the input array for an NFFT operation. The returned tuple has `D` entries. 
Note that this will be the output array for an adjoint NFFT.
"""
@mustimplement size_in(p::AbstractNFFTPlan{T,D,R}) where {T,D,R}

"""
    size_in(p)

Size of the input array for an NFCT operation. The returned tuple has `D` entries. 
Note that this will be the output array for a transposed NFCT.
"""
@mustimplement size_in(p::AbstractNFCTPlan{T,D,R}) where {T,D,R}

"""
    size_out(p)

Size of the output array for an NFFT operation. The returned tuple has `R` entries. 
Note that this will be the input array for an adjoint NFFT.
"""
@mustimplement size_out(p::AbstractNFFTPlan{T,D,R}) where {T,D,R}

"""
    size_out(p)

Size of the output array for an NFCT operation. The returned tuple has `R` entries. 
Note that this will be the input array for a transposed NFCT.
"""
@mustimplement size_out(p::AbstractNFCTPlan{T,D,R}) where {T,D,R}

"""
    nodes!(p, x) -> p

Change nodes `x` in the plan `p` operation and return the plan.
"""
@mustimplement nodes!(p::AbstractNFFTPlan{T}, x::Matrix{T}) where {T}
@mustimplement nodes!(p::AbstractNFCTPlan{T}, x::Matrix{T}) where {T}


## Optional Interface ##
# The following methods can but don't need to be implemented 

@mustimplement apodization!(p::AbstractNFFTPlan, f::AbstractArray, g::AbstractArray)

@mustimplement apodization_adjoint!(p::AbstractNFFTPlan, g::AbstractArray, f::AbstractArray)

@mustimplement convolve!(p::AbstractNFFTPlan, g::AbstractArray, fHat::AbstractArray)

@mustimplement convolve_adjoint!(p::AbstractNFFTPlan, fHat::AbstractArray, g::AbstractArray)



