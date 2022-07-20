
mutable struct NDFTPlan{T,D} <: AbstractNFFTPlan{T,D,1}
  N::NTuple{D,Int64}
  J::Int64
  k::Matrix{T}
end

AbstractNFFTs.size_in(p::NDFTPlan) = p.N
AbstractNFFTs.size_out(p::NDFTPlan) = (p.J,)

mutable struct NDCTPlan{T,D} <: AbstractNFCTPlan{T,D,1}
  N::NTuple{D,Int64}
  J::Int64
  k::Matrix{T}
end

AbstractNFFTs.size_in(p::NDCTPlan) = p.N
AbstractNFFTs.size_out(p::NDCTPlan) = (p.J,)

mutable struct NDSTPlan{T,D} <: AbstractNFSTPlan{T,D,1}
  N::NTuple{D,Int64}
  J::Int64
  k::Matrix{T}
end

AbstractNFFTs.size_in(p::NDSTPlan) = p.N .- 1
AbstractNFFTs.size_out(p::NDSTPlan) = (p.J,)

mutable struct NNDFTPlan{T} <: AbstractNNFFTPlan{T,1,1}
  N::Int64
  J::Int64
  k::Matrix{T}
  y::Matrix{T}
end

AbstractNFFTs.size_in(p::NNDFTPlan) = (p.N,)
AbstractNFFTs.size_out(p::NNDFTPlan) = (p.J,)

### constructors ###

function NDFTPlan(k::Matrix{T}, N::NTuple{D,Int}; kwargs...) where {T,D}

  if D != size(k,1)
    throw(ArgumentError("Nodes x have dimension $(size(k,1)) != $D"))
  end

  J = size(k, 2)

  return NDFTPlan{T,D}(N, J, k)
end

function NDCTPlan(k::Matrix{T}, N::NTuple{D,Int}; kwargs...) where {T,D}

  if D != size(k,1)
    throw(ArgumentError("Nodes x have dimension $(size(k,1)) != $D"))
  end

  J = size(k, 2)

  return NDCTPlan{T,D}(N, J, k)
end

function NDSTPlan(k::Matrix{T}, N::NTuple{D,Int}; kwargs...) where {T,D}

  if D != size(k,1)
    throw(ArgumentError("Nodes x have dimension $(size(k,1)) != $D"))
  end

  J = size(k, 2)

  return NDSTPlan{T,D}(N, J, k)
end

function NNDFTPlan(k::Matrix{T}, y::Matrix{T}; kwargs...) where {T}

  J = size(k, 2)
  N = size(y, 2)

  return NNDFTPlan{T}(N, J, k, y)
end

### ndft functions ###

ndft(k::AbstractArray, f::AbstractArray; kargs...) =
    NDFTPlan(k, size(f); kargs...) * f

ndft_adjoint(k, N, fHat::AbstractVector; kargs...) =
    adjoint(NDFTPlan(k, N; kargs...)) * fHat

ndct(k::AbstractArray, f::AbstractArray; kargs...) =
    NDCTPlan(k, size(f); kargs...) * f

ndct_transposed(k, N, fHat::AbstractVector; kargs...) =
    transpose(NDCTPlan(k, N; kargs...)) * fHat




function LinearAlgebra.mul!(g::AbstractArray{Tg}, plan::NDFTPlan{Tp,D}, f::AbstractArray{T,D}) where {D,Tp,T,Tg}

    plan.N == size(f) ||
        throw(DimensionMismatch("Data f is not consistent with NDFTPlan"))
    plan.J== length(g) ||
        throw(DimensionMismatch("Output g is inconsistent with NDFTPlan"))

    g .= zero(Tg)

    for l=1:prod(plan.N)
        idx = CartesianIndices(plan.N)[l]

        for j=1:plan.J
            arg = zero(T)
            for d=1:D
                arg += plan.k[d,j] * ( idx[d] - 1 - plan.N[d] / 2 )
            end
            g[j] += f[l] * cis(-2*pi*arg)
        end
    end

    return g
end


function LinearAlgebra.mul!(g::AbstractArray{Tg,D}, pl::Adjoint{Complex{Tp},<:NDFTPlan{Tp,D}}, fHat::AbstractVector{T}) where {D,Tp,T,Tg}
  p = pl.parent

  p.J== length(fHat) ||
      throw(DimensionMismatch("Data f inconsistent with NDFTPlan"))
  p.N == size(g) ||
      throw(DimensionMismatch("Output g inconsistent with NDFTPlan"))

  g .= zero(Tg)

  for l=1:prod(p.N)
      idx = CartesianIndices(p.N)[l]

      for j=1:p.J
          arg = zero(T)
          for d=1:D
              arg += p.k[d,j] * ( idx[d] - 1 - p.N[d] / 2 )
          end
          g[l] += fHat[j] * cis(2*pi*arg)
      end
  end

  return g
end

function LinearAlgebra.mul!(g::AbstractArray{Tg}, p::NDCTPlan{Tp,D}, f::AbstractArray{T,D}) where {D,Tp,T,Tg}

    p.N == size(f) ||
        throw(DimensionMismatch("Data f is not consistent with NDCTPlan"))
    p.J== length(g) ||
        throw(DimensionMismatch("Output g is inconsistent with NDCTPlan"))

    g .= zero(Tg)

    for l=1:prod(p.N)
        idx = CartesianIndices(p.N)[l]

        for j=1:p.J
            arg = one(T)
            for d=1:D
                arg *= cos( 2 * pi * p.k[d,j] * ( idx[d] - 1 ) )
            end
            g[j] += f[l] * arg
        end
    end

    return g
end


function LinearAlgebra.mul!(g::AbstractArray{Tg,D}, pl::Transpose{Tp,<:NDCTPlan{Tp,D}}, fHat::AbstractVector{T}) where {D,Tp,T,Tg}
  p = pl.parent

  p.J== length(fHat) ||
      throw(DimensionMismatch("Data f inconsistent with NDCTPlan"))
  p.N == size(g) ||
      throw(DimensionMismatch("Output g inconsistent with NDCTPlan"))

  g .= zero(Tg)

  for l=1:prod(p.N)
      idx = CartesianIndices(p.N)[l]

      for j=1:p.J
          arg = one(T)
          for d=1:D
              arg *= cos( 2 * pi * p.k[d,j] * ( idx[d] - 1 ) )
          end
          g[l] += fHat[j] * arg
      end
  end

  return g
end


function LinearAlgebra.mul!(g::AbstractArray{Tg}, p::NDSTPlan{Tp,D}, f::AbstractArray{T,D}) where {D,Tp,T,Tg}

    p.N == size(f) .+ 1 ||
        throw(DimensionMismatch("Data f is not consistent with NDSTPlan"))
    p.J== length(g) ||
        throw(DimensionMismatch("Output g is inconsistent with NDSTPlan"))

    g .= zero(Tg)

    for l=1:prod(p.N .- 1)
        idx = CartesianIndices(p.N .- 1)[l]

        for j=1:p.J
            arg = one(T)
            for d=1:D
                arg *= sin( 2 * pi * p.k[d,j] * ( idx[d] ) )
            end
            g[j] += f[l] * arg
        end
    end

    return g
end


function LinearAlgebra.mul!(g::AbstractArray{Tg,D}, pl::Transpose{Tp,<:NDSTPlan{Tp,D}}, fHat::AbstractVector{T}) where {D,Tp,T,Tg}
  p = pl.parent

  p.J== length(fHat) ||
      throw(DimensionMismatch("Data f inconsistent with NDSTPlan"))
  p.N == size(g) .+ 1 ||
      throw(DimensionMismatch("Output g inconsistent with NDSTPlan"))

  g .= zero(Tg)

  for l=1:prod(p.N .- 1)
      idx = CartesianIndices(p.N .- 1)[l]

      for j=1:p.J
          arg = one(T)
          for d=1:D
              arg *= sin( 2 * pi * p.k[d,j] * ( idx[d] ) )
          end
          g[l] += fHat[j] * arg
      end
  end

  return g
end


function LinearAlgebra.mul!(g::AbstractArray{Tg}, p::NNDFTPlan{Tp}, f::AbstractArray{T}) where {Tp,T,Tg}

  g .= zero(Tg)

  for l=1:p.N
      for j=1:p.J
          arg = zero(T)
          for d=1:size(p.k,1)
              arg += p.k[d,j] * p.y[d,l]
          end
          g[j] += f[l] * cis(-2*pi*arg)
      end
  end

  return g
end

function LinearAlgebra.mul!(g::AbstractArray{Tg}, pl::Adjoint{Complex{Tp},<:NNDFTPlan{Tp}}, fHat::AbstractVector{T}) where {Tp,T,Tg}
  p = pl.parent
  g .= zero(Tg)

  for l=1:p.N
      for j=1:p.J
          arg = zero(T)
          for d=1:size(p.k,1)
            arg += p.k[d,j] * p.y[d,l]
          end
          g[l] += fHat[j] * cis(2*pi*arg)
      end
  end

  return g
end


