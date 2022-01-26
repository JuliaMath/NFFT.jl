import FINUFFT

mutable struct FINUFFTPlan{T,D} <: AbstractNFFTPlan{T,D,1} 
  N::NTuple{D,Int64}
  M::Int64
  x::Matrix{T}
  m::Int
  σ::T
  reltol::T
  fftflags::UInt32
end

################
# constructors
################

function FINUFFTPlan(x::Matrix{T}, N::NTuple{D,Int}; 
              dims::Union{Integer,UnitRange{Int64}}=1:D,
              fftflags=UInt32(NFFT.FFTW.ESTIMATE), 
              kargs...) where {D,T}

  if dims != 1:D
    error("FINUFFT directional plans not yet implemented!")
  end

  M = size(x,2)

  m, σ, reltol = accuracyParams(; kargs...)

  reltol = max(reltol, 1.0e-15)

  return FINUFFTPlan(N, M, x * 2π, m, T(σ), reltol, fftflags)
end


function Base.show(io::IO, p::FINUFFTPlan{D}) where {D}
  print(io, "FINUFFTPlan")
end

AbstractNFFTs.size_in(p::FINUFFTPlan) = Int.(p.N)
AbstractNFFTs.size_out(p::FINUFFTPlan) = (Int(p.M),)

function AbstractNFFTs.nfft!(p::FINUFFTPlan{T,1}, f::AbstractArray, fHat::StridedArray;
             verbose=false, timing::Union{Nothing,TimingStats} = nothing) where {T}

  FINUFFT.nufft1d2!(p.x, fHat, -1, p.reltol, f, 
                    nthreads = Threads.nthreads(), fftw = p.fftflags)

  return fHat
end

function AbstractNFFTs.nfft_adjoint!(p::FINUFFTPlan{T,1}, fHat::AbstractArray, f::StridedArray;
                     verbose=false, timing::Union{Nothing,TimingStats} = nothing) where {T}

  FINUFFT.nufft1d1!(p.x, fHat, 1, p.reltol, f,
                    nthreads = Threads.nthreads(), fftw = p.fftflags)

  return f
end

function AbstractNFFTs.nfft!(p::FINUFFTPlan{T,2}, f::AbstractArray, fHat::StridedArray;
  verbose=false, timing::Union{Nothing,TimingStats} = nothing) where {T}


  FINUFFT.nufft2d2!(p.x[1,:], p.x[2,:], fHat, -1, p.reltol, f, 
                    nthreads = Threads.nthreads(), fftw = p.fftflags)

  return fHat
end

function AbstractNFFTs.nfft_adjoint!(p::FINUFFTPlan{T,2}, fHat::AbstractArray, f::StridedArray;
          verbose=false, timing::Union{Nothing,TimingStats} = nothing) where {T}

  FINUFFT.nufft2d1!(p.x[1,:], p.x[2,:], fHat, 1, p.reltol, f, 
                    nthreads = Threads.nthreads(), fftw = p.fftflags)

  return f
end

function AbstractNFFTs.nfft!(p::FINUFFTPlan{T,3}, f::AbstractArray, fHat::StridedArray;
  verbose=false, timing::Union{Nothing,TimingStats} = nothing) where {T}

  FINUFFT.nufft3d2!(p.x[1,:], p.x[2,:], p.x[3,:], fHat, -1, p.reltol, f, 
                    nthreads = Threads.nthreads(), fftw = p.fftflags)

  return fHat
end

function AbstractNFFTs.nfft_adjoint!(p::FINUFFTPlan{T,3}, fHat::AbstractArray, f::StridedArray;
          verbose=false, timing::Union{Nothing,TimingStats} = nothing) where {T}

  FINUFFT.nufft3d1!(p.x[1,:], p.x[2,:], p.x[3,:], fHat, 1, p.reltol, f, 
                    nthreads = Threads.nthreads(), fftw = p.fftflags)

  return f
end

