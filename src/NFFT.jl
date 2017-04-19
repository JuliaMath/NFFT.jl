module NFFT

import Base.ind2sub
using Base.Cartesian

export NFFTPlan, nfft, nfft_adjoint, ndft, ndft_adjoint, nfft_performance

include("windowFunctions.jl")

#=
Some internal documentation (especially for people familiar with the nfft)

- Currently the window cannot be changed and defaults to the kaiser-bessel
  window. This is done for simplicity and due to the fact that the
  kaiser-bessel window usually outperforms any other window

- The window is precomputed during construction of the NFFT plan
  When performing the nfft convolution, the LUT of the window is used to
  perform linear interpolation. This approach is reasonable fast and does not
  require too much memory. There are, however alternatives known that are either
  faster or require no extra memory at all.

The non-exported functions apodization and convolve are implemented
using Cartesian macros, that may not be very readable.
This is a conscious decision where performance has outweighed readability.
More readable versions can be (and have been) written using the CartesianRange approach,
but at the time of writing this approach require *a lot* of memory.
=#


#=
D is the number of dimensions of the array to be transformed.
DIM is the dimension along which the array is transformed.
DIM == 0 is the ordinary NFFT, i.e., where all dimensions are transformed.
DIM is a type parameter since it allows the @generated macro to
compile more efficient methods.
=#
type NFFTPlan{D,DIM,T}
  N::NTuple{D,Int64}
  M::Int64
  x::Matrix{T}
  m::Int64
  sigma::T
  n::NTuple{D,Int64}
  K::Int64
  windowLUT::Vector{Vector{T}}
  windowHatInvLUT::Vector{Vector{T}}
  forwardFFT::Base.DFT.FFTW.cFFTWPlan{Complex{Float64},-1,true,D}
  backwardFFT::Base.DFT.FFTW.cFFTWPlan{Complex{Float64},1,true,D}
  tmpVec::Array{Complex{T},D}
end

@inline dim{D,DIM}(::NFFTPlan{D,DIM}) = DIM

@doc """
	NFFTPlan(x, N, ...) -> plan

Compute `D` dimensional NFFT plan for sampling locations `x` (a vector or a `D`-by-`M` matrix) that can be applied on arrays of size `N` (a tuple of length `D`).

The optional arguments control the accuracy.
"""->
function NFFTPlan{D,T}(x::AbstractMatrix{T}, N::NTuple{D,Int}, m=4, sigma=2.0,
                       window=:kaiser_bessel, K=2000)
  if !isa(x, Matrix)
	  x = collect(x)
  end

  if D != size(x,1)
    throw(ArgumentError())
  end

  n = ntuple(d->round(Int,sigma*N[d]), D)

  tmpVec = zeros(Complex{T}, n)

  M = size(x,2)

  FP = plan_fft!(tmpVec)
  BP = plan_bfft!(tmpVec)

  # Create lookup table
  win, win_hat = getWindow(window)

  windowLUT = Vector{Vector{T}}(D)
  Z = round(Int,3*K/2)
  for d=1:D
    windowLUT[d] = zeros(T, Z)
    for l=1:Z
      y = ((l-1) / (K-1)) * m/n[d]
      windowLUT[d][l] = win(y, n[d], m, sigma)
    end
  end

  windowHatInvLUT = Vector{Vector{T}}(D)
  for d=1:D
    windowHatInvLUT[d] = zeros(T, N[d])
    for k=1:N[d]
      windowHatInvLUT[d][k] = 1. / win_hat(k-1-N[d]/2, n[d], m, sigma)
    end
  end

  NFFTPlan{D,0,T}(N, M, x, m, sigma, n, K, windowLUT, windowHatInvLUT, FP, BP, tmpVec )
end

function NFFTPlan(x::AbstractVector, N::Integer, m=4, sigma=2.0, window=:kaiser_bessel, K=2000)
  NFFTPlan(reshape(x,1,length(x)), (N,), m, sigma, window, K)
end


# Directional NFFT
@doc """
	NFFTPlan(x, d, N, ...) -> plan

Compute *directional* NFFT plan:
A 1D plan that is applied along dimension `d` of a `D` dimensional array of size `N` with sampling locations `x` (a vector).
"""->
function NFFTPlan{D,T}(x::AbstractVector{T}, dim::Integer, N::NTuple{D,Int64}, m=4,
                       sigma=2.0, window=:kaiser_bessel, K=2000)
  n = ntuple(d->round(Int, sigma*N[d]), D)

  sz = [N...]
  sz[dim] = n[dim]
  tmpVec = Array{Complex{T}}(sz...)

  M = length(x)

  FP = plan_fft!(tmpVec, dim)
  BP = plan_bfft!(tmpVec, dim)

  # Create lookup table
  win, win_hat = getWindow(window)

  windowLUT = Vector{Vector{T}}(1)
  Z = round(Int, 3*K/2)
  windowLUT[1] = zeros(T, Z)
  for l = 1:Z
	  y = ((l-1) / (K-1)) * m/n[dim]
	  windowLUT[1][l] = win(y, n[dim], m, sigma)
  end

  windowHatInvLUT = Vector{Vector{T}}(1)
  windowHatInvLUT[1] = zeros(T, N[dim])
  for k = 1:N[dim]
	  windowHatInvLUT[1][k] = 1. / win_hat(k-1-N[dim]/2, n[dim], m, sigma)
  end

  NFFTPlan{D,dim,T}(N, M, reshape(x,1,M), m, sigma, n, K, windowLUT, windowHatInvLUT, FP, BP, tmpVec)
end

function NFFTPlan{D,T}(x::Matrix{T}, dim::Integer, N::NTuple{D,Int}, m=4, sigma=2.0, window=:kaiser_bessel, K=2000)
  if size(x,1) != 1 && size(x,2) != 1
	  throw(DimensionMismatch())
  end

  NFFTPlan(vec(x), dim, N, m, sigma, window, K)
end


function Base.show{D}(io::IO, p::NFFTPlan{D,0})
	print(io, "NFFTPlan with ", p.M, " sampling points for ", p.N, " array")
end

function Base.show{D,DIM}(io::IO, p::NFFTPlan{D,DIM})
	print(io, "NFFTPlan with ", p.M, " sampling points for ", p.N, " array along dimension ", DIM)
end


@generated function consistencyCheck{D,DIM,T}(p::NFFTPlan{D,DIM}, f::AbstractArray{T,D}, fHat::AbstractArray{T})
	quote
        if $DIM == 0
            fHat_test = (p.M == length(fHat))
        elseif $DIM > 0
            fHat_test = @nall $D d -> ( d == $DIM ? size(fHat,d) == p.M : size(fHat,d) == p.N[d] )
        end

		if p.N != size(f) || !fHat_test
			throw(DimensionMismatch("Data is not consistent with NFFTPlan"))
		end
	end
end


### nfft functions ###

@doc """
	nfft!(p, f, fHat) -> fHat

Calculate the NFFT of `f` with plan `p` and store the result in `fHat`.

Both `f` and `fHat` must be complex arrays.
"""->
function nfft!{T}(p::NFFTPlan, f::AbstractArray{T}, fHat::StridedArray{T})
  consistencyCheck(p, f, fHat)

  fill!(p.tmpVec, zero(T))
  @inbounds apodization!(p, f, p.tmpVec)
  if nprocs() == 1
    p.forwardFFT * p.tmpVec # fft!(p.tmpVec) or fft!(p.tmpVec, dim)
  else
    dim(p) == 0 ? fft!(p.tmpVec) : fft!(p.tmpVec, dim(p))
  end
  @inbounds convolve!(p, p.tmpVec, fHat)
  return fHat
end

@doc """
	nfft(p, f) -> fHat

For a **non**-directional `D` dimensional plan `p` this calculates the NFFT of a `D` dimensional array `f` of size `N`.
`fHat` is a vector of length `M`.
(`M` and `N` are defined in the plan creation)

For a **directional** `D` dimensional plan `p` both `f` and `fHat` are `D`
dimensional arrays, and the dimension specified in the plan creation is
affected.
"""->
function nfft{D,T}(p::NFFTPlan{D,0}, f::AbstractArray{T,D})
  fHat = zeros(T, p.M)
  nfft!(p, f, fHat)
  return fHat
end

function nfft{D,T}(x, f::AbstractArray{T,D})
  p = NFFTPlan(x, size(f) )
  return nfft(p, f)
end

function nfft{D,DIM,T}(p::NFFTPlan{D,DIM}, f::AbstractArray{T,D})
  sz = [p.N...]
  sz[DIM] = p.M
  fHat = Array{T}(sz...)
  nfft!(p, f, fHat)
  return fHat
end


@doc """
	nfft_adjoint!(p, fHat, f) -> f

Calculate the adjoint NFFT of `fHat` and store the result in `f`.

Both `f` and `fHat` must be complex arrays.
"""->
function nfft_adjoint!(p::NFFTPlan, fHat::AbstractArray, f::StridedArray)
  consistencyCheck(p, f, fHat)

  @inbounds convolve_adjoint!(p, fHat, p.tmpVec)
  if nprocs() == 1
    p.backwardFFT * p.tmpVec # bfft!(p.tmpVec) or bfft!(p.tmpVec, dim)
  else
    dim(p) == 0 ? bfft!(p.tmpVec) : bfft!(p.tmpVec, dim(p))
  end
  @inbounds apodization_adjoint!(p, p.tmpVec, f)
  return f
end

@doc """
	nfft_adjoint(p, f) -> fHat

For a **non**-directional `D` dimensional plan `p` this calculates the adjoint NFFT of a length `M` vector `fHat`
`f` is a `D` dimensional array of size `N`.
(`M` and `N` are defined in the plan creation)

For a **directional** `D` dimensional plan `p` both `f` and `fHat` are `D`
dimensional arrays, and the dimension specified in the plan creation is
affected.
"""->
function nfft_adjoint{D,DIM,T}(p::NFFTPlan{D,DIM}, fHat::AbstractArray{T})
  f = Array{T}(p.N)
  nfft_adjoint!(p, fHat, f)
  return f
end

function nfft_adjoint{D,T}(x, N::NTuple{D,Int}, fHat::AbstractVector{T})
  p = NFFTPlan(x, N)
  return nfft_adjoint(p, fHat)
end


### ndft functions ###

# fallback for 1D
function ind2sub{T}(::Array{T,1}, idx::Int)
  idx
end

function ndft{T,D}(plan::NFFTPlan{D}, f::AbstractArray{T,D})
  plan.N == size(f) || throw(DimensionMismatch("Data is not consistent with NFFTPlan"))

  g = zeros(T, plan.M)

  for l=1:prod(plan.N)
    idx = ind2sub(plan.N,l)

    for k=1:plan.M
      arg = zero(T)
      for d=1:D
        arg += plan.x[d,k] * ( idx[d] - 1 - plan.N[d] / 2 )
      end
      g[k] += f[l] * cis(-2*pi*arg)
    end
  end

  return g
end

function ndft_adjoint{T,D}(plan::NFFTPlan{D}, fHat::AbstractArray{T,1})
  plan.M == length(fHat) || throw(DimensionMismatch("Data is not consistent with NFFTPlan"))

  g = zeros(T, plan.N)

  for l=1:prod(plan.N)
    idx = ind2sub(plan.N,l)

    for k=1:plan.M
      arg = zero(T)
      for d=1:D
        arg += plan.x[d,k] * ( idx[d] - 1 - plan.N[d] / 2 )
      end
      g[l] += fHat[k] * cis(2*pi*arg)
    end
  end

  return g
end



### convolve! ###

function convolve!{T}(p::NFFTPlan{1,0}, g::AbstractVector{T}, fHat::StridedVector{T})
  fill!(fHat, zero(T))
  scale = 1.0 / p.m * (p.K-1)
  n = p.n[1]

  for k=1:p.M # loop over nonequispaced nodes
    c = floor(Int, p.x[k]*n)
    for l=(c-p.m):(c+p.m) # loop over nonzero elements
      gidx = rem(l+n, n) + 1
      idx = abs( (p.x[k]*n - l)*scale ) + 1
      idxL = floor(Int, idx)

      fHat[k] += g[gidx] * (p.windowLUT[1][idxL] + ( idx-idxL ) * (p.windowLUT[1][idxL+1] - p.windowLUT[1][idxL]))
    end
  end
end

function convolve!{D,T}(p::NFFTPlan{D,0}, g::AbstractArray{T,D}, fHat::StridedVector{T})
    scale = 1.0 / p.m * (p.K-1)

    Threads.@threads for k in 1:p.M
        fHat[k] = _convolve(p, g, scale, k)
    end
end


@generated function _convolve{D,T}(p::NFFTPlan{D,0}, g::AbstractArray{T,D}, scale, k)
	quote
        @nexprs $D d -> xscale_d = p.x[d,k] * p.n[d]
        @nexprs $D d -> c_d = floor(Int, xscale_d)

        fHat = zero(T)

        @nloops $D l d -> (c_d-p.m):(c_d+p.m) d->begin
            # preexpr
            gidx_d = rem(l_d+p.n[d], p.n[d]) + 1
            idx = abs( (xscale_d - l_d)*scale ) + 1
            idxL = floor(idx)
            idxInt = Int(idxL)
            tmpWin_d = p.windowLUT[d][idxInt] + ( idx-idxL ) * (p.windowLUT[d][idxInt+1] - p.windowLUT[d][idxInt])
        end begin
            # bodyexpr
            v = @nref $D g gidx
            @nexprs $D d -> v *= tmpWin_d
            fHat += v
        end

        return fHat
    end
end

@generated function convolve!{D,DIM,T}(p::NFFTPlan{D,DIM}, g::AbstractArray{T,D}, fHat::StridedArray{T,D})
	quote
		fill!(fHat, zero(T))
		scale = 1.0 / p.m * (p.K-1)

		for k in 1:p.M
			xscale = p.x[k] * p.n[$DIM]
			c = floor(Int, xscale)
			@nloops $D l d->begin
				# rangeexpr
				if d == $DIM
					(c-p.m):(c+p.m)
				else
					1:size(g,d)
				end
			end d->begin
				# preexpr
				if d == $DIM
					gidx_d = rem(l_d+p.n[d], p.n[d]) + 1
					idx = abs( (xscale - l_d)*scale ) + 1
					idxL = floor(idx)
					idxInt = Int(idxL)
					tmpWin = p.windowLUT[1][idxInt] + ( idx-idxL ) * (p.windowLUT[1][idxInt+1] - p.windowLUT[1][idxInt])
					fidx_d = k
				else
					gidx_d = l_d
					fidx_d = l_d
				end
			end begin
				# bodyexpr
				(@nref $D fHat fidx) += (@nref $D g gidx) * tmpWin
			end
		end
	end
end


### convolve_adjoint! ###

function convolve_adjoint!{T}(p::NFFTPlan{1,0}, fHat::AbstractVector{T}, g::StridedVector{T})
  fill!(g, zero(T))
  scale = 1.0 / p.m * (p.K-1)
  n = p.n[1]

  for k=1:p.M # loop over nonequispaced nodes
    c = round(Int,p.x[k]*n)
    for l=(c-p.m):(c+p.m) # loop over nonzero elements
      gidx = rem(l+n, n) + 1
      idx = abs( (p.x[k]*n - l)*scale ) + 1
      idxL = round(Int, idx)

      g[gidx] += fHat[k] * (p.windowLUT[1][idxL] + ( idx-idxL ) * (p.windowLUT[1][idxL+1] - p.windowLUT[1][idxL]))
    end
  end
end

@generated function convolve_adjoint!{D,T}(p::NFFTPlan{D,0}, fHat::AbstractVector{T}, g::StridedArray{T,D})
	quote
		fill!(g, zero(T))
		scale = 1.0 / p.m * (p.K-1)

		for k in 1:p.M
			@nexprs $D d -> xscale_d = p.x[d,k] * p.n[d]
			@nexprs $D d -> c_d = floor(Int, xscale_d)

			@nloops $D l d -> (c_d-p.m):(c_d+p.m) d->begin
				# preexpr
				gidx_d = rem(l_d+p.n[d], p.n[d]) + 1
				idx = abs( (xscale_d - l_d)*scale ) + 1
				idxL = floor(idx)
				idxInt = Int(idxL)
				tmpWin_d = p.windowLUT[d][idxInt] + ( idx-idxL ) * (p.windowLUT[d][idxInt+1] - p.windowLUT[d][idxInt])
			end begin
				# bodyexpr
				v = fHat[k]
				@nexprs $D d -> v *= tmpWin_d
				(@nref $D g gidx) += v
			end
		end
	end
end

@generated function convolve_adjoint!{D,DIM,T}(p::NFFTPlan{D,DIM}, fHat::AbstractArray{T,D}, g::StridedArray{T,D})
	quote
		fill!(g, zero(T))
		scale = 1.0 / p.m * (p.K-1)

		for k in 1:p.M
			xscale = p.x[k] * p.n[$DIM]
			c = floor(Int, xscale)
			@nloops $D l d->begin
				# rangeexpr
				if d == $DIM
					(c-p.m):(c+p.m)
				else
					1:size(g,d)
				end
			end d->begin
				# preexpr
				if d == $DIM
					gidx_d = rem(l_d+p.n[d], p.n[d]) + 1
					idx = abs( (xscale - l_d)*scale ) + 1
					idxL = floor(idx)
					idxInt = Int(idxL)
					tmpWin = p.windowLUT[1][idxInt] + ( idx-idxL ) * (p.windowLUT[1][idxInt+1] - p.windowLUT[1][idxInt])
					fidx_d = k
				else
					gidx_d = l_d
					fidx_d = l_d
				end
			end begin
				# bodyexpr
				(@nref $D g gidx) += (@nref $D fHat fidx) * tmpWin
			end
		end
	end
end


### apodization! ###

function apodization!{T}(p::NFFTPlan{1,0}, f::AbstractVector{T}, g::StridedVector{T})
  n = p.n[1]
  N = p.N[1]
  const offset = round( Int, n - N / 2 ) - 1
  for l=1:N
    g[((l+offset)% n) + 1] = f[l] * p.windowHatInvLUT[1][l]
  end
end

@generated function apodization!{D,T}(p::NFFTPlan{D,0}, f::AbstractArray{T,D}, g::StridedArray{T,D})
	quote
		@nexprs $D d -> offset_d = round(Int, p.n[d] - p.N[d]/2) - 1

		@nloops $D l f d->(gidx_d = rem(l_d+offset_d, p.n[d]) + 1) begin
			v = @nref $D f l
			@nexprs $D d -> v *= p.windowHatInvLUT[d][l_d]
			(@nref $D g gidx) = v
		end
	end
end

@generated function apodization!{D,DIM,T}(p::NFFTPlan{D,DIM}, f::AbstractArray{T,D}, g::StridedArray{T,D})
	quote
		offset = round(Int, p.n[$DIM] - p.N[$DIM]/2) - 1

		@nloops $D l f d->begin
			# preexpr
			if d == $DIM
				gidx_d = rem(l_d+offset, p.n[d]) + 1
				winidx = l_d
			else
				gidx_d = l_d
			end
		end begin
			# bodyexpr
			(@nref $D g gidx) = (@nref $D f l) * p.windowHatInvLUT[1][winidx]
		end
	end
end


### apodization_adjoint! ###

function apodization_adjoint!{T}(p::NFFTPlan{1,0}, g::AbstractVector{T}, f::StridedVector{T})
  n = p.n[1]
  N = p.N[1]
  const offset = round( Int, n - N / 2 ) - 1
  for l=1:N
    f[l] = g[((l+offset)% n) + 1] * p.windowHatInvLUT[1][l]
  end
end

@generated function apodization_adjoint!{T,D}(p::NFFTPlan{D,0}, g::AbstractArray{T,D}, f::StridedArray{T,D})
	quote
		@nexprs $D d -> offset_d = round(Int, p.n[d] - p.N[d]/2) - 1

		@nloops $D l f begin
			v = @nref $D g d -> rem(l_d+offset_d, p.n[d]) + 1
			@nexprs $D d -> v *= p.windowHatInvLUT[d][l_d]
			(@nref $D f l) = v
		end
	end
end

@generated function apodization_adjoint!{D,DIM,T}(p::NFFTPlan{D,DIM}, g::AbstractArray{T,D}, f::StridedArray{T,D})
	quote
		offset = round(Int, p.n[$DIM] - p.N[$DIM]/2) - 1

		@nloops $D l f d->begin
			# preexpr
			if d == $DIM
				gidx_d = rem(l_d+offset, p.n[d]) + 1
				winidx = l_d
			else
				gidx_d = l_d
			end
		end begin
			# bodyexpr
			(@nref $D f l) = (@nref $D g gidx) * p.windowHatInvLUT[1][winidx]
		end
	end
end


### performance test ###

function nfft_performance()

  m = 4
  sigma = 2.0

  # 1D

  N = 2^19
  M = N

  x = rand(M) .- 0.5
  fHat = rand(M)*1im

  println("NFFT Performance Test 1D")

  tic()
  p = NFFTPlan(x,N,m,sigma)
  println("initialization")
  toc()

  tic()
  fApprox = nfft_adjoint(p,fHat)
  println("adjoint")
  toc()

  tic()
  fHat2 = nfft(p, fApprox);
  println("trafo")
  toc()

  N = 1024
  M = N*N

  x2 = rand(2,M) .- 0.5
  fHat = rand(M)*1im

  println("NFFT Performance Test 2D")

  tic()
  p = NFFTPlan(x2,(N,N),m,sigma)
  println("initialization")
  toc()

  tic()
  fApprox = nfft_adjoint(p,fHat)
  println("adjoint")
  toc()

  tic()
  fHat2 = nfft(p, fApprox);
  println("trafo")
  toc()

  N = 32
  M = N*N*N

  x3 = rand(3,M) .- 0.5
  fHat = rand(M)*1im

  println("NFFT Performance Test 3D")

  tic()
  p = NFFTPlan(x3,(N,N,N),m,sigma)
  println("initialization")
  toc()

  tic()
  fApprox = nfft_adjoint(p,fHat)
  println("adjoint")
  toc()

  tic()
  fHat2 = nfft(p, fApprox);
  println("trafo")
  toc()

end

include("samplingDensity.jl")

end
