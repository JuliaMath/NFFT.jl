"""
  AbstractNFFTPlan{T,D,R}

Abstract type for an NFFT plan.
* T is the element type (Float32/Float64)
* D is the number of dimensions of the input array.
* R is the number of dimensions of the output array.
"""
abstract type AbstractNFFTPlan{T,D,R} end


@enum PrecomputeFlags begin
  LUT = 1
  FULL = 2
  TENSOR = 3
  FULL_LUT = 4
end



#####################
# define interface
#####################

"""
    nfft!(p, f, fHat) -> fHat

Inplace NFFT transforming the `D` dimensional array `f` to the `R` dimensional array `fHat`.
The transformation is applied along `D-R+1` dimensions specified in the plan `p`.
Both `f` and `fHat` must be complex arrays of element type `Complex{T}`.
"""
@mustimplement nfft!(p::AbstractNFFTPlan{T,D,R}, f::AbstractArray, fHat::AbstractArray) where {T,D,R}

"""
    nfft_adjoint!(p, fHat, f) -> f

Inplace adjoint NFFT transforming the `R` dimensional array `fHat` to the `D` dimensional array `f`.
The transformation is applied along `D-R+1` dimensions specified in the plan `p`.
Both `f` and `fHat` must be complex arrays of element type `Complex{T}`.
"""
@mustimplement nfft_adjoint!(p::AbstractNFFTPlan{T,D,R}, fHat::AbstractArray, f::AbstractArray) where {T,D,R}

"""
    size_in(p)

Size of the input array for an `nfft!` operation. The returned tuple has `D` entries. 
Note that this will be the output array for `nfft_adjoint!`.
"""
@mustimplement size_in(p::AbstractNFFTPlan{T,D,R}) where {T,D,R}

"""
    size_out(p)

Size of the output array for an `nfft!` operation. The returned tuple has `R` entries. 
Note that this will be the input array for `nfft_adjoint!`.
"""
@mustimplement size_out(p::AbstractNFFTPlan{T,D,R}) where {T,D,R}
