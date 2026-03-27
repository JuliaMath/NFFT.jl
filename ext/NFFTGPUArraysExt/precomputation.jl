function NFFT.precomputeB(win, k::AbstractGPUArray, N::NTuple{D,Int}, Ñ::NTuple{D,Int}, m, J, σ, K, T) where D
  I = similar(k, Int64, (2*m)^D, J)
  β = (2*m)^D

  # CPU uses a CSC constructor, which is not generically available for GPU (I think)
  #Y = similar(k, Int64, J + 1)
  #Y .= (0:J) .* β .+ 1
  # We have to use the COO constructor and need (2*m)^D * J values:
  Y = similar(k, Int64, (2*m)^D * J)
  Y .= ((0:β*J-1) .÷ β) .+ 1
  
  V = similar(k, T, (2*m)^D, J)
  nProd = ntuple(d-> (d==1) ? 1 : prod(Ñ[1:(d-1)]), D)
  L = Val(2*m)

  @kernel inbounds = true function precomputeB_kernel(I, V, win, k, Ñ::NTuple{D,Int}, m, σ, nProd, ::Val{Z}) where {D, Z}
    idx = @index(Global, Cartesian)
    j = idx[2]
    linear = idx[1]
    
    prodWin = one(eltype(k))
    ζ = 1
    tmpIdx = linear - 1 # 0-based for index calculation
    @unroll for d = 1:D
      l_d = (tmpIdx % Z) + 1 # index in 1:(2*m)
      tmpIdx = div(tmpIdx, Z)

      kscale = k[d, j] * Ñ[d]
      off = floor(Int, kscale) - m + 1

      idx_d = rem(l_d + off + Ñ[d] - 1, Ñ[d]) + 1 # periodic wrapped index in 1:Ñ[d] 
      ζ += (idx_d - 1) * nProd[d]

      # accumulate window product
      prodWin *= win( (kscale - (l_d-1) - off)  / Ñ[d], Ñ[d], m, σ)
    end

    I[idx] = ζ
    V[idx] = prodWin
  end

  backend = get_backend(k)
  kernel = precomputeB_kernel(backend)
  kernel(I, V, win, k, Ñ, m, σ, nProd, L, ndrange = size(I))

  S = sparse(vec(I), Y, vec(V), prod(Ñ), J)
  return S
end