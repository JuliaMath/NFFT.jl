# CuNNFTPlan2d
mutable struct CuNFFTPlan{T,D} <: AbstractNFFTPlan{T,D,1}
  N::NTuple{D,Int64}
  M::Int64
  x::Array{T,2}
  m::Int64
  sigma::T
  n::NTuple{D,Int64}
  K::Int64
  windowHatInvLUT::CuArray{Complex{T},D}
  forwardFFT::CUDA.CUFFT.cCuFFTPlan{Complex{T},-1,true,D} 
  backwardFFT::CUDA.CUFFT.cCuFFTPlan{Complex{T},1,true,D}
  tmpVec::CuArray{Complex{T},D}
  tmpVec2::CuArray{Complex{T},D}
  apodIdx::CuArray{Int64,1}
  B::CuSparseMatrixCSC{Complex{T}}
end

function AbstractNFFTs.plan_nfft(::Type{CuArray}, x::Matrix{T}, N::NTuple{D,Int}, rest...;
  timing::Union{Nothing,TimingStats} = nothing, kargs...) where {T,D}
  t = @elapsed begin
    p = CuNFFTPlan(x, N, rest...; kargs...)
  end
  if timing != nothing
    timing.pre = t
  end
  return p
end

# constructor for CuNFFTPlan2d
function CuNFFTPlan(x::Matrix{T}, N::NTuple{D,Int}, m=4, sigma=2.0,
  window=:kaiser_bessel, K=2000; kwargs...) where {T,D} 

  if D != size(x,1)
    throw(ArgumentError("Nodes x have dimension $(size(x,1)) != $D!"))
  end

  if any(isodd.(N))
    throw(ArgumentError("N = $N needs to consist of even integers!"))
  end

  n = ntuple(d->(ceil(Int,sigma*N[d])÷2)*2, D) # ensure that n is an even integer
  sigma = n[1] / N[1]

  tmpVec = CuArray(zeros(Complex{T},n))
  tmpVec2 = CuArray(zeros(Complex{T},N))
  M = size(x,2)

  # create FFT-plans
  dims = ntuple(d->d, D)
  FP = plan_fft!(tmpVec, dims)
  BP = plan_bfft!(tmpVec, dims)

  # Create lookup table for 1d interpolation kernels
  win, win_hat = NFFT.getWindow(window)
  windowHatInvLUT  = precomp_windowHatInvLUT(T, win_hat, N, sigma, m)
  windowHatInvLUT_d = CuArray(windowHatInvLUT)

  # compute (linear) indices for apodization (mapping from regular to oversampled grid)
  apodIdx = precomp_apodIdx(N,n)
  apodIdx_d = CuArray(apodIdx)

  # precompute interpolation matrix
  U1 = ntuple(d-> (d==1) ? 1 : (2*m+1)^(d-1), D)
  U2 = ntuple(d-> (d==1) ? 1 : prod(n[1:(d-1)]), D)
  B = precomputeB(win, x, n, m, M, sigma, T, U1, U2)
  B_d = CuSparseMatrixCSC(Complex{T}.(B))

  return CuNFFTPlan{T,D}(N, M, x, m, sigma, n, K, windowHatInvLUT_d,
                  FP, BP, tmpVec, tmpVec2, apodIdx_d, B_d )
end

AbstractNFFTs.size_in(p::CuNFFTPlan) = p.N
AbstractNFFTs.size_out(p::CuNFFTPlan) = (p.M,)

"""  in-place NFFT on the GPU"""
function AbstractNFFTs.nfft!(p::CuNFFTPlan{T,D}, f::CuArray, fHat::CuArray) where {T,D} 
  consistencyCheck(p, f, fHat)

  # apodization
  fill!(p.tmpVec, zero(Complex{T}))
  p.tmpVec2 .= f.*p.windowHatInvLUT
  p.tmpVec[p.apodIdx] = p.tmpVec2

  # FFT
  p.forwardFFT * p.tmpVec # equivalent to fft!(p.tmpVec)

  # convolution
  mul!(fHat, transpose(p.B), vec(p.tmpVec)) 

  return fHat
end

"""  in-place adjoint NFFT on the GPU"""
function AbstractNFFTs.nfft_adjoint!(p::CuNFFTPlan{T,D}, fHat::CuArray, f::CuArray) where {T,D}
  consistencyCheck(p, f, fHat)

  # adjoint convolution
  mul!(vec(p.tmpVec), p.B, fHat)

  # FFT
  p.backwardFFT * p.tmpVec # bfft!(p.tmpVec)

  # adjoint apodization
  p.tmpVec2[:] = p.tmpVec[p.apodIdx]
  f .= p.tmpVec2.*p.windowHatInvLUT
  
  return f
end
