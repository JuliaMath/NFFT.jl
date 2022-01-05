"""
  AbstractNFFTPlan{T,D,R}
"""
abstract type AbstractNFFTPlan{T,D,R} end

#####################
# define interface
#####################

# in-place NFFT
@mustimplement nfft!(p::AbstractNFFTPlan{T,D,R}, f::AbstractArray, fHat::AbstractArray) where {T,D,R}
# in-place adjoint NFFT
@mustimplement nfft_adjoint!(p::AbstractNFFTPlan{T,D,R}, fHat::AbstractArray, f::AbstractArray) where {T,D,R}
# size of the input array for the NFFT
@mustimplement size_in(p::AbstractNFFTPlan{T,D,R}) where {T,D,R}
# size of the output array for the NFFT
@mustimplement size_out(p::AbstractNFFTPlan{T,D,R}) where {T,D,R}
