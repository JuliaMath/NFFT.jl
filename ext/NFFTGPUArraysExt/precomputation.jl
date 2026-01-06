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

  @kernel cpu = false inbounds = true function precomputeB_kernel(I, V, win, k, Ñ::NTuple{D,Int}, m, σ, nProd, ::Val{Z}) where {D, Z}
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

#=
@inline @generated function _precomputeB(win, k::AbstractMatrix{T}, N::NTuple{D,Int}, Ñ::NTuple{D,Int}, m, J,
                     σ, scale, I, Y, V, mProd, nProd, L::Val{Z}, j, LUTSize) where {T, 3, Z}
  quote

    (tmpIdx_1, tmpWin_1) = precomputeOneNode(win, k, Ñ, m, σ, scale, j, 1, L, LUTSize)
    (tmpIdx_2, tmpWin_2) = precomputeOneNode(win, k, Ñ, m, σ, scale, j, 2, L, LUTSize)
    (tmpIdx_3, tmpWin_3) = precomputeOneNode(win, k, Ñ, m, σ, scale, j, 3, L, LUTSize)


    κ_3 = 1
    ζ_3 = 1
    prodWin_3 = one(T)

    for l_3 in 1:Z
        prodWin_2 = prodWin_3 * tmpWin_3[l_3]
        κ_2 = κ_3 + (l_3 - 1) * mProd[3]
        ζ_2 = ζ_3 + (tmpIdx_3[l_3] - 1) * nProd[3]

        for l_2 in 1:Z
            prodWin_1 = prodWin_2 * tmpWin_2[l_2]
            κ_1 = κ_2 + (l_2 - 1) * mProd[2]
            ζ_1 = ζ_2 + (tmpIdx_2[l_2] - 1) * nProd[2]

            for l_1 in 1:Z
                prodWin_0 = prodWin_1 * tmpWin_1[l_1]
                κ_0 = κ_1 + (l_1 - 1) * mProd[1]
                ζ_0 = ζ_1 + (tmpIdx_1[l_1] - 1) * nProd[1]

                I[κ_0, j] = ζ_0
                V[κ_0, j] = prodWin_0
            end
        end
    end
    return
  end
end

@inline function _precomputeB(win, k::AbstractMatrix{T}, N::NTuple{1,Int}, Ñ::NTuple{1,Int}, m, J,
                              σ, scale, I, Y, V, mProd, nProd, L::Val{Z}, j, LUTSize) where {T, Z}

  # Precompute per-dimension lookup
  tmpIdx_1, tmpWin_1 = precomputeOneNode(win, k, Ñ, m, σ, scale, j, 1, L, LUTSize)

  # Initial values (equivalent to κ_1 = 1, ζ_1 = 1, prodWin_1 = one(T))
  κ_1 = 1
  ζ_1 = 1
  prodWin_1 = one(T)

  for l_1 in 1:Z
    prodWin_0 = prodWin_1 * tmpWin_1[l_1]
    κ_0 = κ_1 + (l_1 - 1) * mProd[1]
    ζ_0 = ζ_1 + (tmpIdx_1[l_1] - 1) * nProd[1]
    I[κ_0, j] = ζ_0
    V[κ_0, j] = prodWin_0
  end
  return
end

@inline function _precomputeB(win, k::AbstractMatrix{T}, N::NTuple{2,Int}, Ñ::NTuple{2,Int}, m, J,
                              σ, scale, I, Y, V, mProd, nProd, L::Val{Z}, j, LUTSize) where {T, Z}

    # Precompute per-dimension lookup
    tmpIdx_1, tmpWin_1 = precomputeOneNode(win, k, Ñ, m, σ, scale, j, 1, L, LUTSize)
    tmpIdx_2, tmpWin_2 = precomputeOneNode(win, k, Ñ, m, σ, scale, j, 2, L, LUTSize)

    # Initial values (equivalent to κ_2 = 1, ζ_2 = 1, prodWin_2 = one(T))
    κ_2 = 1
    ζ_2 = 1
    prodWin_2 = one(T)

    for l_2 in 1:Z
      prodWin_1 = prodWin_2 * tmpWin_2[l_2]
      κ_1 = κ_2 + (l_2 - 1) * mProd[2]
      ζ_1 = ζ_2 + (tmpIdx_2[l_2] - 1) * nProd[2]

      for l_1 in 1:Z
        prodWin_0 = prodWin_1 * tmpWin_1[l_1]
        κ_0 = κ_1 + (l_1 - 1) * mProd[1]
        ζ_0 = ζ_1 + (tmpIdx_1[l_1] - 1) * nProd[1]

        I[κ_0, j] = ζ_0
        V[κ_0, j] = prodWin_0
      end
    end
    return
end


### precomputation of the window and the indices required during convolution ###

@inline function precomputeOneNode(win::Function, k::AbstractMatrix{T}, Ñ::NTuple{1,Int}, m,
                                   σ, scale, j, d, L::Val{Z}, LUTSize) where {T,Z}
    kscale = k[d, j] * Ñ[d]
    off = floor(Int, kscale) - m + 1
    tmpIdx = ntuple(l -> rem(l + off + Ñ[d] - 1, Ñ[d]) + 1, Z)
    tmpWin = ntuple(l -> win((kscale - (l - 1) - off) / Ñ[d], Ñ[d], m, σ), Z)
    return tmpIdx, tmpWin
end
=#