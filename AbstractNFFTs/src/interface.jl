"""
  AbstractFTPlan{T,D,R}

Abstract type for any NFFT-like plan (NFFT, NFFT, NFCT, NFST).
* T is the element type (Float32/Float64)
* D is the number of dimensions of the input array.
* R is the number of dimensions of the output array.
"""
abstract type AbstractFTPlan{T,D,R} end

"""
  AbstractRealFTPlan{T,D,R}

Abstract type for either an NFCT plan or an NFST plan.
* T is the element type (Float32/Float64)
* D is the number of dimensions of the input array.
* R is the number of dimensions of the output array.
"""
abstract type AbstractRealFTPlan{T,D,R} <: AbstractFTPlan{T,D,R} end

"""
  AbstractComplexFTPlan{T,D,R}

Abstract type for either an NFFT plan or an NNFFT plan.
* T is the element type (Float32/Float64)
* D is the number of dimensions of the input array.
* R is the number of dimensions of the output array.
"""
abstract type AbstractComplexFTPlan{T,D,R} <: AbstractFTPlan{T,D,R} end


"""
  AbstractNFFTPlan{T,D,R}

Abstract type for an NFFT plan.
* T is the element type (Float32/Float64)
* D is the number of dimensions of the input array.
* R is the number of dimensions of the output array.
"""
abstract type AbstractNFFTPlan{T,D,R} <: AbstractComplexFTPlan{T,D,R} end

"""
  AbstractNFCTPlan{T,D,R}

Abstract type for an NFCT plan.
* T is the element type (Float32/Float64)
* D is the number of dimensions of the input array.
* R is the number of dimensions of the output array.
"""
abstract type AbstractNFCTPlan{T,D,R} <: AbstractRealFTPlan{T,D,R} end

"""
  AbstractNFSTPlan{T,D,R}

Abstract type for an NFST plan.
* T is the element type (Float32/Float64)
* D is the number of dimensions of the input array.
* R is the number of dimensions of the output array.
"""
abstract type AbstractNFSTPlan{T,D,R} <: AbstractRealFTPlan{T,D,R} end

"""
  AbstractNNFFTPlan{T,D,R}

Abstract type for an NNFFT plan.
* T is the element type (Float32/Float64)
* D is the number of dimensions of the input array.
* R is the number of dimensions of the output array.
"""
abstract type AbstractNNFFTPlan{T,D,R} <: AbstractComplexFTPlan{T,D,R} end

#####################
# Function needed to make AbstractFTPlan an operator 
#####################

Base.eltype(p::AbstractComplexFTPlan{T}) where T = Complex{T}
Base.eltype(p::AbstractRealFTPlan{T}) where T = T

Base.size(p::AbstractFTPlan) = (prod(size_out(p)), prod(size_in(p)))
Base.length(p::AbstractFTPlan) = prod(size(p))

Base.size(p::Adjoint{Complex{T}, U}) where {T, U<:AbstractComplexFTPlan{T}} = 
  (prod(size_in(p.parent)), prod(size_out(p.parent)))

Base.size(p::Transpose{T, U}) where {T, U<:AbstractRealFTPlan{T}} = 
  (prod(size_in(p.parent)), prod(size_out(p.parent)))

LinearAlgebra.adjoint(p::AbstractComplexFTPlan{T,D,R}) where {T,D,R} = 
  Adjoint{Complex{T}, typeof(p)}(p)

LinearAlgebra.transpose(p::AbstractRealFTPlan{T,D,R}) where {T,D,R} = 
  Transpose{T, typeof(p)}(p)

size_in(p::Adjoint{Complex{T},<:AbstractComplexFTPlan{T,D,R}}) where {T,D,R} = size_out(p.parent)
size_out(p::Adjoint{Complex{T},<:AbstractComplexFTPlan{T,D,R}}) where {T,D,R} = size_in(p.parent)

size_in(p::Transpose{T,<:AbstractRealFTPlan{T,D,R}}) where {T,D,R} = size_out(p.parent)
size_out(p::Transpose{T,<:AbstractRealFTPlan{T,D,R}}) where {T,D,R} = size_in(p.parent)

# The following are required both, because there seem to be special array handling in base
function Base.show(io::IO, mime::MIME{Symbol("text/plain")}, p::Transpose{T,<:AbstractRealFTPlan{T,D,R}}) where {T,D,R}
  print(io, "Transposed ")
  print(io, p.parent)
end

function Base.show(io::IO, p::Transpose{T,<:AbstractRealFTPlan{T,D,R}}) where {T,D,R}
  print(io, "Transposed ")
  print(io, p.parent)
end

function Base.show(io::IO, p::Adjoint{Complex{T},<:AbstractComplexFTPlan{T,D,R}}) where {T,D,R}
  print(io, "Adjoint ")
  print(io, p.parent)
end

function Base.show(io::IO, mime::MIME{Symbol("text/plain")}, p::Adjoint{Complex{T},<:AbstractComplexFTPlan{T,D,R}}) where {T,D,R}
  print(io, "Adjoint ")
  print(io, p.parent)
end

function Base.copy(p::Transpose{T,<:AbstractRealFTPlan{T,D,R}}) where {T,D,R}
  transpose(copy(p.parent))
end

function Base.copy(p::Adjoint{Complex{T},<:AbstractComplexFTPlan{T,D,R}}) where {T,D,R}
  adjoint(copy(p.parent))
end

#####################
# define interface
#####################


"""
    mul!(fHat, p, f) -> fHat

Inplace NFFT/NFCT/NFST/NNFFT transforming the `D` dimensional array `f` to the `R` dimensional array `fHat`.
The transformation is applied along `D-R+1` dimensions specified in the plan `p`.
"""
@mustimplement LinearAlgebra.mul!( fHat::AbstractArray, p::AbstractFTPlan{T}, f::AbstractArray) where {T}

"""
    mul!(f, p, fHat) -> f

Inplace adjoint NFFT/NNFFT transforming the `R` dimensional array `fHat` to the `D` dimensional array `f`.
The transformation is applied along `D-R+1` dimensions specified in the plan `p`.
"""
@mustimplement LinearAlgebra.mul!(f::AbstractArray, p::Adjoint{Complex{T},<:AbstractComplexFTPlan{T}}, fHat::AbstractArray) where {T}

"""
    mul!(f, p, fHat) -> f

Inplace transposed NFCT/NFST transforming the `R` dimensional array `fHat` to the `D` dimensional array `f`.
The transformation is applied along `D-R+1` dimensions specified in the plan `p`.
"""
@mustimplement LinearAlgebra.mul!(f::AbstractArray, p::Transpose{T,<:AbstractRealFTPlan{T}}, fHat::AbstractArray) where {T}

"""
    size_in(p)

Size of the input array for the plan p (NFFT/NFCT/NFST/NNFFT). 
The returned tuple has `R` entries. 
Note that this will be the output size for the transposed / adjoint operator.
"""
@mustimplement size_in(p::AbstractFTPlan{T,D,R}) where {T,D,R}

"""
    size_out(p)

Size of the output array for the plan p (NFFT/NFCT/NFST/NNFFT). 
The returned tuple has `R` entries. 
Note that this will be the input size for the transposed / adjoint operator.
"""
@mustimplement size_out(p::AbstractFTPlan{T,D,R}) where {T,D,R}

"""
    nodes!(p, k) -> p

Change nodes `k` in the plan `p` operation and return the plan.
"""
@mustimplement nodes!(p::AbstractFTPlan{T}, k::Matrix{T}) where {T}


## Optional Interface ##
# The following methods can but don't need to be implemented 

@mustimplement deconvolve!(p::AbstractNFFTPlan, f::AbstractArray, g::AbstractArray)
@mustimplement deconvolve_transpose!(p::AbstractNFFTPlan, g::AbstractArray, f::AbstractArray)
@mustimplement convolve!(p::AbstractNFFTPlan, g::AbstractArray, fHat::AbstractArray)
@mustimplement convolve_transpose!(p::AbstractNFFTPlan, fHat::AbstractArray, g::AbstractArray)



