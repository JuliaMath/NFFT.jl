
mutable struct NDFTPlan{T,D} <: AbstractNFFTPlan{T,D,1}
  N::NTuple{D,Int64}
  M::Int64
  x::Matrix{T}
end

AbstractNFFTs.size_in(p::NDFTPlan) = p.N
AbstractNFFTs.size_out(p::NDFTPlan) = (p.M,)

mutable struct NDCTPlan{T,D} <: AbstractNFCTPlan{T,D,1}
  N::NTuple{D,Int64}
  M::Int64
  x::Matrix{T}
end

AbstractNFFTs.size_in(p::NDCTPlan) = p.N
AbstractNFFTs.size_out(p::NDCTPlan) = (p.M,)

mutable struct NNDFTPlan{T} <: AbstractNNFFTPlan{T,1,1}
  N::Int64
  M::Int64
  x::Matrix{T}
  y::Matrix{T}
end

AbstractNFFTs.size_in(p::NNDFTPlan) = (p.N,)
AbstractNFFTs.size_out(p::NNDFTPlan) = (p.M,)

### constructors ###

function NDFTPlan(x::Matrix{T}, N::NTuple{D,Int}; kwargs...) where {T,D}

  if D != size(x,1)
    throw(ArgumentError("Nodes x have dimension $(size(x,1)) != $D"))
  end

  M = size(x, 2)

  return NDFTPlan{T,D}(N, M, x)
end

function NDCTPlan(x::Matrix{T}, N::NTuple{D,Int}; kwargs...) where {T,D}

  if D != size(x,1)
    throw(ArgumentError("Nodes x have dimension $(size(x,1)) != $D"))
  end

  M = size(x, 2)

  return NDCTPlan{T,D}(N, M, x)
end

function NNDFTPlan(x::Matrix{T}, y::Matrix{T}; kwargs...) where {T}

  M = size(x, 2)
  N = size(y, 2)

  return NNDFTPlan{T}(N, M, x, y)
end

### ndft functions ###

ndft(x::AbstractArray, f::AbstractArray; kargs...) =
    NDFTPlan(x, size(f); kargs...) * f

ndft_adjoint(x, N, fHat::AbstractVector; kargs...) =
    adjoint(NDFTPlan(x, N; kargs...)) * fHat

ndct(x::AbstractArray, f::AbstractArray; kargs...) =
    NDCTPlan(x, size(f); kargs...) * f

ndct_transposed(x, N, fHat::AbstractVector; kargs...) =
    transpose(NDCTPlan(x, N; kargs...)) * fHat




function LinearAlgebra.mul!(g::AbstractArray{Tg}, plan::NDFTPlan{Tp,D}, f::AbstractArray{T,D}) where {D,Tp,T,Tg}

    plan.N == size(f) ||
        throw(DimensionMismatch("Data f is not consistent with NDFTPlan"))
    plan.M == length(g) ||
        throw(DimensionMismatch("Output g is inconsistent with NDFTPlan"))

    g .= zero(Tg)

    for l=1:prod(plan.N)
        idx = CartesianIndices(plan.N)[l]

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


function LinearAlgebra.mul!(g::AbstractArray{Tg,D}, pl::Adjoint{Complex{Tp},<:NDFTPlan{Tp,D}}, fHat::AbstractVector{T}) where {D,Tp,T,Tg}
  p = pl.parent

  p.M == length(fHat) ||
      throw(DimensionMismatch("Data f inconsistent with NDFTPlan"))
  p.N == size(g) ||
      throw(DimensionMismatch("Output g inconsistent with NDFTPlan"))

  g .= zero(Tg)

  for l=1:prod(p.N)
      idx = CartesianIndices(p.N)[l]

      for k=1:p.M
          arg = zero(T)
          for d=1:D
              arg += p.x[d,k] * ( idx[d] - 1 - p.N[d] / 2 )
          end
          g[l] += fHat[k] * cis(2*pi*arg)
      end
  end

  return g
end

function LinearAlgebra.mul!(g::AbstractArray{Tg}, plan::NDCTPlan{Tp,D}, f::AbstractArray{T,D}) where {D,Tp,T,Tg}

    plan.N == size(f) ||
        throw(DimensionMismatch("Data f is not consistent with NDFTPlan"))
    plan.M == length(g) ||
        throw(DimensionMismatch("Output g is inconsistent with NDFTPlan"))

    g .= zero(Tg)

    for l=1:prod(plan.N)
        idx = CartesianIndices(plan.N)[l]

        for k=1:plan.M
            arg = one(T)
            for d=1:D
                arg *= cos( 2 * pi * plan.x[d,k] * ( idx[d] - 1 ) )
            end
            g[k] += f[l] * arg
        end
    end

    return g
end


function LinearAlgebra.mul!(g::AbstractArray{Tg,D}, pl::Transpose{Tp,<:NDCTPlan{Tp,D}}, fHat::AbstractVector{T}) where {D,Tp,T,Tg}
  p = pl.parent

  p.M == length(fHat) ||
      throw(DimensionMismatch("Data f inconsistent with NDFTPlan"))
  p.N == size(g) ||
      throw(DimensionMismatch("Output g inconsistent with NDFTPlan"))

  g .= zero(Tg)

  for l=1:prod(p.N)
      idx = CartesianIndices(p.N)[l]

      for k=1:p.M
          arg = one(T)
          for d=1:D
              arg *= cos( 2 * pi * plan.x[d,k] * ( idx[d] - 1 ) )
          end
          g[l] += fHat[k] * arg
      end
  end

  return g
end


function LinearAlgebra.mul!(g::AbstractArray{Tg}, plan::NNDFTPlan{Tp}, f::AbstractArray{T}) where {Tp,T,Tg}

  g .= zero(Tg)

  for l=1:plan.N
      for k=1:plan.M
          arg = zero(T)
          for d=1:size(plan.x,1)
              arg += plan.x[d,k] * plan.y[d,l]
          end
          g[k] += f[l] * cis(-2*pi*arg)
      end
  end

  return g
end

function LinearAlgebra.mul!(g::AbstractArray{Tg}, pl::Adjoint{Complex{Tp},<:NNDFTPlan{Tp}}, fHat::AbstractVector{T}) where {Tp,T,Tg}
  p = pl.parent
  g .= zero(Tg)

  for l=1:p.N
      for k=1:p.M
          arg = zero(T)
          for d=1:size(p.x,1)
            arg += p.x[d,k] * p.y[d,l]
          end
          g[l] += fHat[k] * cis(2*pi*arg)
      end
  end

  return g
end


