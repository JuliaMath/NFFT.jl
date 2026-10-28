abstract type AbstractNFFTBackend end
struct BackendReference
    ref::Base.RefValue{Union{Missing, AbstractNFFTBackend}}
    BackendReference(val::Union{Missing, AbstractNFFTBackend}) = new(Ref{Union{Missing, AbstractNFFTBackend}}(val))
end
Base.setindex!(ref::BackendReference, val::Union{Missing, AbstractNFFTBackend}) = ref.ref[] = val
Base.setindex!(ref::BackendReference, val::Module) = setindex!(ref, val.backend())
Base.getindex(ref::BackendReference) = getindex(ref.ref)::Union{Missing, AbstractNFFTBackend}
Base.convert(::Type{BackendReference}, val::AbstractNFFTBackend) = BackendReference(val)
const nfft_backend = ScopedValue(BackendReference(missing))

"""
    set_active_backend!(back::Union{Missing, Module, AbstractNFFTBackend})

Set the default NFFT plan backend. A module `back` must implement `back.backend()`.
"""
set_active_backend!(back::Module) = set_active_backend!(back.backend())
function set_active_backend!(back::Union{Missing, AbstractNFFTBackend})
  nfft_backend[][] = back
end
active_backend() = nfft_backend[][]
function no_backend_error() 
  error(
    """
    No default backend available!
    Make sure to also "import/using" an NFFT backend such as NFFT or NonuniformFFTs.
    """
  )
end

"""
    AbstractFTPlan{T,D,R}

Abstract type for any NFFT-like plan (NFFT, NNFFT, RNFFT, NFCT, NFST).
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

"""
    AbstractRNFFTPlan{T,D,R}

Abstract type for an NFFT plan with real values at the nodes (the NFFT counterpart of FFTW's
`brfft`/`rfft`): the direct transform maps the half `n₁ ≥ 0` of the complex coefficients to
real values at the nodes, the adjoint maps real values at the nodes to that half.
* T is the real element type (Float32/Float64)
* D is the number of dimensions of the input array.
* R is the number of dimensions of the output array.
"""
abstract type AbstractRNFFTPlan{T,D,R} <: AbstractFTPlan{T,D,R} end

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

# The direct RNFFT takes the real part, so it is linear over the reals only; its adjoint is
# taken with respect to the real inner product Re⟨x, y⟩.
LinearAlgebra.adjoint(p::AbstractRNFFTPlan{T,D,R}) where {T,D,R} =
  Adjoint{Complex{T}, typeof(p)}(p)
Base.size(p::Adjoint{Complex{T}, U}) where {T, U<:AbstractRNFFTPlan{T}} =
  (prod(size_in(p.parent)), prod(size_out(p.parent)))
size_in(p::Adjoint{Complex{T},<:AbstractRNFFTPlan{T,D,R}}) where {T,D,R} = size_out(p.parent)
size_out(p::Adjoint{Complex{T},<:AbstractRNFFTPlan{T,D,R}}) where {T,D,R} = size_in(p.parent)
Base.copy(p::Adjoint{Complex{T},<:AbstractRNFFTPlan{T,D,R}}) where {T,D,R} = adjoint(copy(p.parent))
Base.show(io::IO, p::Adjoint{Complex{T},<:AbstractRNFFTPlan{T,D,R}}) where {T,D,R} =
  (print(io, "Adjoint "); print(io, p.parent))

size_in(p::Adjoint{Complex{T},<:AbstractComplexFTPlan{T,D,R}}) where {T,D,R} = size_out(p.parent)
size_out(p::Adjoint{Complex{T},<:AbstractComplexFTPlan{T,D,R}}) where {T,D,R} = size_in(p.parent)

size_in(p::Transpose{T,<:AbstractRealFTPlan{T,D,R}}) where {T,D,R} = size_out(p.parent)
size_out(p::Transpose{T,<:AbstractRealFTPlan{T,D,R}}) where {T,D,R} = size_in(p.parent)

# Transpose and Adjoint are AbstractMatrix: without the text/plain method, the REPL prints them as arrays.
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

In-place transform of `f` (size `size_in(p)`) into `fHat` (size `size_out(p)`).
"""
@mustimplement LinearAlgebra.mul!( fHat::AbstractArray, p::AbstractFTPlan{T}, f::AbstractArray) where {T}

"""
    mul!(f, p, fHat) -> f

In-place adjoint transform of `fHat` (size `size_out(p)`) into `f` (size `size_in(p)`).
"""
@mustimplement LinearAlgebra.mul!(f::AbstractArray, p::Adjoint{Complex{T},<:AbstractComplexFTPlan{T}}, fHat::AbstractArray) where {T}

"""
    mul!(f, p, fHat) -> f

In-place transposed NFCT/NFST of `fHat` (size `size_out(p)`) into `f` (size `size_in(p)`).
"""
@mustimplement LinearAlgebra.mul!(f::AbstractArray, p::Transpose{T,<:AbstractRealFTPlan{T}}, fHat::AbstractArray) where {T}

"""
    size_in(p)

Size of the input array for the plan p (NFFT/NFCT/NFST/NNFFT). 
The returned tuple has `D` entries.
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
@mustimplement nodes!(p::AbstractFTPlan, k::AbstractMatrix)


"""
    nodes(p) -> k

Nodes of the plan `p` (a d × J matrix).
"""
@mustimplement nodes(p::AbstractFTPlan)

"""
    accuracy(p) -> Float64

Relative error (ℓ² norm over random data) the plan `p` is expected to achieve; what the
requested `reltol` turned into.
"""
@mustimplement accuracy(p::AbstractFTPlan)

## Optional interface ##
# Stage interface for implementations built from the three steps of Knopp et al. 2023:
# resampling correction → FFT → resampling (direct), and the adjoint steps in reverse order.
# The resampling correction writes the oversampled spectrum `ĝ` of size `spectrum_size(p)`, the FFT
# turns it into the equidistant oversampled array `g` of size `oversampled_size(p)`. For NFFT, NFCT
# and NFST plans both have the same size and the FFT stages also work in place; for NFCT and NFST
# plans they are the DCT-I/DST-I and its transpose, and the adjoint steps are transposes. For RNFFT
# plans `ĝ` is the complex half spectrum of the real array `g`, and the FFT stages only exist out of
# place.

"""
    oversampled_size(p) -> NTuple

Size of the oversampled equidistant array the FFT of plan `p` acts on. The stage interface is
defined for plans without batch dimensions; an implementation throws an `ArgumentError` for others.
"""
@mustimplement oversampled_size(p::AbstractFTPlan)

"""
    spectrum_size(p) -> NTuple

Size of the oversampled spectrum `ĝ` that the resampling correction writes and the FFT stage
reads: `oversampled_size(p)`, except for RNFFT plans (the half spectrum `(Ñ₁÷2+1, Ñ₂, …)`).
"""
spectrum_size(p::AbstractFTPlan) = oversampled_size(p)

"""
    resampling_correction!(p, f, ĝ) -> ĝ

First step of the direct NFFT: divide `f` by the Fourier transform of the window function and
place it into the zero-padded oversampled spectrum `ĝ`.
"""
@mustimplement resampling_correction!(p::AbstractFTPlan, f::AbstractArray, ĝ::AbstractArray)

"""
    oversampled_fft!(p, g) -> g
    oversampled_fft!(p, ĝ, g) -> g

Second step of the direct NFFT: the FFT of the oversampled spectrum, in place on `g` or from `ĝ`
into `g`. The out-of-place form leaves `ĝ` unchanged.
"""
@mustimplement oversampled_fft!(p::AbstractFTPlan, g::AbstractArray)
oversampled_fft!(p::AbstractFTPlan, ĝ::AbstractArray, g::AbstractArray) = oversampled_fft!(p, copyto!(g, ĝ))

"""
    resampling!(p, g, fHat) -> fHat

Third step of the direct NFFT: resample the oversampled equidistant array `g` at the nodes.
"""
@mustimplement resampling!(p::AbstractFTPlan, g::AbstractArray, fHat::AbstractArray)

"""
    resampling_adjoint!(p, fHat, g) -> g

First step of the adjoint NFFT: the adjoint resampling of the values `fHat` at the nodes onto
the oversampled equidistant array `g` (overwritten).
"""
@mustimplement resampling_adjoint!(p::AbstractFTPlan, fHat::AbstractArray, g::AbstractArray)

"""
    oversampled_bfft!(p, g) -> g
    oversampled_bfft!(p, g, ĝ) -> ĝ

Second step of the adjoint NFFT: the adjoint (unnormalised backward) FFT, in place on `g` or from
`g` into `ĝ`. The out-of-place form leaves `g` unchanged.
"""
@mustimplement oversampled_bfft!(p::AbstractFTPlan, g::AbstractArray)
oversampled_bfft!(p::AbstractFTPlan, g::AbstractArray, ĝ::AbstractArray) = oversampled_bfft!(p, copyto!(ĝ, g))

"""
    resampling_correction_adjoint!(p, ĝ, f) -> f

Third step of the adjoint NFFT: crop the oversampled spectrum `ĝ` and divide by the Fourier
transform of the window function.
"""
@mustimplement resampling_correction_adjoint!(p::AbstractFTPlan, ĝ::AbstractArray, f::AbstractArray)
