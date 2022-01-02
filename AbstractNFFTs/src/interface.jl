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
