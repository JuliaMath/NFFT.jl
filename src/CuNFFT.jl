export CuNFFTPlan

"""CuNNFTPlan2d"""
mutable struct CuNFFTPlan{T,D,DN} 
  N::NTuple{D,Int64}
  M::Int64
  x::Array{T,2}
  numSlices::Int64
  m::Int64
  sigma::T
  n::NTuple{D,Int64}
  K::Int64
  windowHatInvLUT::CuArray{Complex{T},D}
  forwardFFT::CUDA.CUFFT.cCuFFTPlan{Complex{T},-1,true,DN} 
  backwardFFT::CUDA.CUFFT.cCuFFTPlan{Complex{T},1,true,DN}
  tmpVec::CuArray{Complex{T},DN}
  tmpVec2::CuArray{Complex{T},DN}
  apodIdx::CuArray{Int64,1}
  B::CuSparseMatrixCSC{Complex{T}}
end

""" constructor for CuNFFTPlan2d"""
function CuNFFTPlan(x::Matrix{T}, N::NTuple{D,Int}; numSlices::Int64=1, m=4, sigma=2.0,
  window=:kaiser_bessel, K=2000, kwargs...) where {T,D} 

  n = ntuple(d->round(Int,sigma*N[d]), D)
  n_cat = Tuple( vcat(collect(n)...,numSlices))
  N_cat = Tuple( vcat(collect(N)...,numSlices))

  tmpVec = CuArray(zeros(Complex{T},n_cat))
  tmpVec2 = CuArray(zeros(Complex{T},N_cat))
  M = size(x,2)

  # create FFT-plans
  # using directional derivatives allows us to transform multiple
  # slices of a higher tensor in parallel
  dims = ntuple(d->d, D)
  FP = plan_fft!(tmpVec, dims; kwargs...)
  BP = plan_bfft!(tmpVec, dims; kwargs...)

  # Create lookup table for 1d interpolation kernels
  win, win_hat = NFFT.getWindow(window)
  windowHatInvLUT  = precomp_windowHatInvLUT(T, win_hat, N, sigma, m)
  windowHatInvLUT_d = CuArray(windowHatInvLUT)

  # compute (linear) indices for apodization (mapping from regular to oversampled grid)
  apodIdx = precomp_apodIdx(N, sigma, numSlices)
  apodIdx_d = CuArray(apodIdx)

  # precompute interpolation matrix
  U1 = ntuple(d-> (d==1) ? 1 : (2*m+1)^(d-1), D)
  U2 = ntuple(d-> (d==1) ? 1 : prod(n[1:(d-1)]), D)
  B = precomputeB(win, x, n, m, M, sigma, T, U1, U2)
  B_d = CuSparseMatrixCSC(Complex{T}.(B))

  return CuNFFTPlan{T,D,D+1}(N, M, x, numSlices, m, sigma, n, K, windowHatInvLUT_d,
                  FP, BP, tmpVec, tmpVec2, apodIdx_d, B_d )

end

"""  in-place NFFT on the GPU"""
function nfft!(p::CuNFFTPlan{T}, f::CuArray, fHat::CuArray) where T 
  # apodization
  fill!(p.tmpVec, zero(Complex{T}))
  p.tmpVec2 .= f.*p.windowHatInvLUT
  p.tmpVec[p.apodIdx] = p.tmpVec2

  # FFT
  p.forwardFFT * p.tmpVec # equivalent to fft!(p.tmpVec)

  # convolution
  mul!(fHat, transpose(p.B), reshape(p.tmpVec,:,p.numSlices)) 

  return fHat
end

function nfft(p::CuNFFTPlan{T,D,DN}, f::CuArray) where {T,D,DN}
  fHat = CuArray{Complex{T}}(undef, p.M, p.numSlices)
  nfft!(p, f, fHat)
  return fHat
end

"""  in-place adjoint NFFT on the GPU"""
function nfft_adjoint!(p::CuNFFTPlan{T}, fHat::CuArray, f::CuArray) where T
  # adjoint convolution
  mul!(reshape(p.tmpVec,:,p.numSlices), p.B, fHat)

  # FFT
  p.backwardFFT * p.tmpVec # bfft!(p.tmpVec)

  # adjoint apodization
  p.tmpVec2[:] = p.tmpVec[p.apodIdx]
  f .= p.tmpVec2.*p.windowHatInvLUT
  
  return f
end

function nfft_adjoint(p::CuNFFTPlan{T,D,DN}, fHat::CuArray{Complex{T},2} ) where {T,D,DN}
  f = CuArray{Complex{T}}(undef, size(p.tmpVec2)) # size(tmpVec2) = (tmpVec2: N[1],...,N[end],numSlices)
  nfft_adjoint!(p, fHat, f)
  return f
end

function nfft_adjoint(p::CuNFFTPlan{T,D,DN}, fHat::CuArray{Complex{T},1} ) where {T,D,DN}
  f = CuArray{Complex{T}}(undef, size(p.tmpVec2)) # size(tmpVec2) = (tmpVec2: N[1],...,N[end],numSlices)
  nfft_adjoint!(p, reshape(fHat,:,1), f)
  f = reshape(f,p.N)
  return f
end