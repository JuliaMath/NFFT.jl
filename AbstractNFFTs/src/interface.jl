"""
  AnyNFFTPlan{T,D,R}

Abstract type for any NFFT-like plan (NFFT, NFFT, NFCT, NFST).
* T is the element type (Float32/Float64)
* D is the number of dimensions of the input array.
* R is the number of dimensions of the output array.
"""
abstract type AnyNFFTPlan{T,D,R} end

"""
  AnyRealNFFTPlan{T,D,R}

Abstract type for either an NFCT plan or an NFST plan.
* T is the element type (Float32/Float64)
* D is the number of dimensions of the input array.
* R is the number of dimensions of the output array.
"""
abstract type AnyRealNFFTPlan{T,D,R} <: AnyNFFTPlan{T,D,R} end

"""
  AnyComplexNFFTPlan{T,D,R}

Abstract type for either an NFFT plan or an NNFFT plan.
* T is the element type (Float32/Float64)
* D is the number of dimensions of the input array.
* R is the number of dimensions of the output array.
"""
abstract type AnyComplexNFFTPlan{T,D,R} <: AnyNFFTPlan{T,D,R} end


"""
  AbstractNFFTPlan{T,D,R}

Abstract type for an NFFT plan.
* T is the element type (Float32/Float64)
* D is the number of dimensions of the input array.
* R is the number of dimensions of the output array.
"""
abstract type AbstractNFFTPlan{T,D,R} <: AnyComplexNFFTPlan{T,D,R} end

"""
  AbstractNFCTPlan{T,D,R}

Abstract type for an NFCT plan.
* T is the element type (Float32/Float64)
* D is the number of dimensions of the input array.
* R is the number of dimensions of the output array.
"""
abstract type AbstractNFCTPlan{T,D,R} <: AnyRealNFFTPlan{T,D,R} end

"""
  AbstractNFSTPlan{T,D,R}

Abstract type for an NFST plan.
* T is the element type (Float32/Float64)
* D is the number of dimensions of the input array.
* R is the number of dimensions of the output array.
"""
abstract type AbstractNFSTPlan{T,D,R} <: AnyRealNFFTPlan{T,D,R} end

"""
  AbstractNNFFTPlan{T,D,R}

Abstract type for an NNFFT plan.
* T is the element type (Float32/Float64)
* D is the number of dimensions of the input array.
* R is the number of dimensions of the output array.
"""
abstract type AbstractNNFFTPlan{T,D,R} <: AnyComplexNFFTPlan{T,D,R} end

#####################
# Function needed to make AnyNFFTPlan an operator 
#####################

Base.eltype(p::AnyComplexNFFTPlan{T}) where T = Complex{T}
Base.eltype(p::AnyRealNFFTPlan{T}) where T = T

Base.size(p::AnyNFFTPlan) = (prod(size_out(p)), prod(size_in(p)))

Base.size(p::Adjoint{Complex{T}, U}) where {T, U<:AnyComplexNFFTPlan{T}} = 
  (prod(size_in(p.parent)), prod(size_out(p.parent)))

Base.size(p::Transpose{T, U}) where {T, U<:AnyRealNFFTPlan{T}} = 
  (prod(size_in(p.parent)), prod(size_out(p.parent)))

LinearAlgebra.adjoint(p::AnyComplexNFFTPlan{T,D,R}) where {T,D,R} = 
  Adjoint{Complex{T}, typeof(p)}(p)

LinearAlgebra.transpose(p::AnyRealNFFTPlan{T,D,R}) where {T,D,R} = 
  Transpose{T, typeof(p)}(p)

size_in(p::Adjoint{Complex{T},<:AnyComplexNFFTPlan{T,D,R}}) where {T,D,R} = size_out(p.parent)
size_out(p::Adjoint{Complex{T},<:AnyComplexNFFTPlan{T,D,R}}) where {T,D,R} = size_in(p.parent)

size_in(p::Transpose{T,<:AnyRealNFFTPlan{T,D,R}}) where {T,D,R} = size_out(p.parent)
size_out(p::Transpose{T,<:AnyRealNFFTPlan{T,D,R}}) where {T,D,R} = size_in(p.parent)

#####################
# define interface
#####################


"""
    mul!(fHat, p, f) -> fHat

Inplace NFFT/NFCT/NFST/NNFFT transforming the `D` dimensional array `f` to the `R` dimensional array `fHat`.
The transformation is applied along `D-R+1` dimensions specified in the plan `p`.
"""
@mustimplement LinearAlgebra.mul!( fHat::AbstractArray, p::AnyNFFTPlan{T}, f::AbstractArray) where {T}

"""
    mul!(f, p, fHat) -> f

Inplace adjoint NFFT/NNFFT transforming the `R` dimensional array `fHat` to the `D` dimensional array `f`.
The transformation is applied along `D-R+1` dimensions specified in the plan `p`.
"""
@mustimplement LinearAlgebra.mul!(f::AbstractArray, p::Adjoint{Complex{T},<:AnyComplexNFFTPlan{T}}, fHat::AbstractArray) where {T}

"""
    mul!(f, p, fHat) -> f

Inplace transposed NFCT/NFST transforming the `R` dimensional array `fHat` to the `D` dimensional array `f`.
The transformation is applied along `D-R+1` dimensions specified in the plan `p`.
"""
@mustimplement LinearAlgebra.mul!(f::AbstractArray, p::Transpose{T,<:AnyRealNFFTPlan{T}}, fHat::AbstractArray) where {T}

"""
    size_in(p)

Size of the input array for the plan p (NFFT/NFCT/NFST/NNFFT). 
The returned tuple has `R` entries. 
Note that this will be the output size for the transposed / adjoint operator.
"""
@mustimplement size_in(p::AnyNFFTPlan{T,D,R}) where {T,D,R}

"""
    size_out(p)

Size of the output array for the plan p (NFFT/NFCT/NFST/NNFFT). 
The returned tuple has `R` entries. 
Note that this will be the input size for the transposed / adjoint operator.
"""
@mustimplement size_out(p::AnyNFFTPlan{T,D,R}) where {T,D,R}

"""
    nodes!(p, x) -> p

Change nodes `x` in the plan `p` operation and return the plan.
"""
@mustimplement nodes!(p::AnyNFFTPlan{T}, x::Matrix{T}) where {T}


## Optional Interface ##
# The following methods can but don't need to be implemented 

@mustimplement apodization!(p::AbstractNFFTPlan, f::AbstractArray, g::AbstractArray)
@mustimplement apodization_adjoint!(p::AbstractNFFTPlan, g::AbstractArray, f::AbstractArray)
@mustimplement convolve!(p::AbstractNFFTPlan, g::AbstractArray, fHat::AbstractArray)
@mustimplement convolve_adjoint!(p::AbstractNFFTPlan, fHat::AbstractArray, g::AbstractArray)



