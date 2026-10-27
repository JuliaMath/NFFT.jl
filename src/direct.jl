"""
    NDFTPlan(k, N)
    NDCTPlan(k, N)
    NDSTPlan(k, N)
    NNDFTPlan(k, y)

Direct evaluation of the NDFT, NDCT, NDST and NNDFT with the definitions of the fast plans
(`O(J·prod(N))` operations), as a reference for tests and small problems. Nodes are a `D × J`
matrix or, for `D = 1`, a vector; the sums are computed in double precision.
"""
struct NDFTPlan{T,D} <: AbstractNFFTPlan{T,D,1}
  N::NTuple{D,Int}
  J::Int
  k::Matrix{T}
end

@doc (@doc NDFTPlan)
struct NDCTPlan{T,D} <: AbstractNFCTPlan{T,D,1}
  N::NTuple{D,Int}
  J::Int
  k::Matrix{T}
end

@doc (@doc NDFTPlan)
struct NDSTPlan{T,D} <: AbstractNFSTPlan{T,D,1}
  N::NTuple{D,Int}
  J::Int
  k::Matrix{T}
end

@doc (@doc NDFTPlan)
struct NNDFTPlan{T} <: AbstractNNFFTPlan{T,1,1}
  N::Int
  J::Int
  k::Matrix{T}
  y::Matrix{T}
end

const DirectPlan = Union{NDFTPlan,NDCTPlan,NDSTPlan}

node_matrix(k::AbstractVector) = reshape(float.(collect(k)), 1, :)
node_matrix(k::AbstractMatrix) = float.(Matrix(k))

for P in (:NDFTPlan, :NDCTPlan, :NDSTPlan)
  @eval function $P(k::AbstractVecOrMat, N::Union{Integer,Dims})
    k = node_matrix(k); N = Tuple(N)
    size(k, 1) == length(N) ||
      throw(ArgumentError("nodes have $(size(k, 1)) dimensions, the size N has $(length(N))"))
    $P{eltype(k),length(N)}(N, size(k, 2), k)
  end
end

function NNDFTPlan(k::AbstractVecOrMat, y::AbstractVecOrMat)
  k, y = node_matrix(k), node_matrix(y)
  size(k, 1) == size(y, 1) ||
    throw(ArgumentError("targets have $(size(k, 1)) dimensions, sources $(size(y, 1))"))
  T = promote_type(eltype(k), eltype(y))
  NNDFTPlan{T}(size(y, 2), size(k, 2), T.(k), T.(y))
end

AbstractNFFTs.size_in(p::DirectPlan) = p.N
AbstractNFFTs.size_out(p::DirectPlan) = (p.J,)
AbstractNFFTs.size_in(p::NNDFTPlan) = (p.N,)
AbstractNFFTs.size_out(p::NNDFTPlan) = (p.J,)
AbstractNFFTs.nodes(p::Union{DirectPlan,NNDFTPlan}) = p.k
AbstractNFFTs.accuracy(p::Union{DirectPlan,NNDFTPlan}) = 0.0

# Matrix entry for node j and input index I (a CartesianIndex, or a source index for the NNDFT).
entry(p::NDFTPlan{T,D}, j, I) where {T,D} =
  cispi(-2 * sum(ntuple(d -> Float64(p.k[d, j]) * (I[d] - 1 - p.N[d] ÷ 2), Val(D))))
entry(p::NDCTPlan{T,D}, j, I) where {T,D} = prod(ntuple(d -> cospi(2 * Float64(p.k[d, j]) * (I[d] - 1)), Val(D)))
entry(p::NDSTPlan{T,D}, j, I) where {T,D} = prod(ntuple(d -> sinpi(2 * Float64(p.k[d, j]) * I[d]), Val(D)))
entry(p::NNDFTPlan, j, l) = cispi(-2 * sum(d -> Float64(p.k[d, j]) * p.y[d, l], axes(p.k, 1)))

indices(p::DirectPlan) = CartesianIndices(p.N)
indices(p::NNDFTPlan) = 1:p.N

function LinearAlgebra.mul!(fHat::AbstractArray, p::Union{DirectPlan,NNDFTPlan}, f::AbstractArray)
  check_size(f, size_in(p), "input"); check_size(fHat, size_out(p), "output")
  for j in 1:p.J
    s = zero(promote_type(eltype(f), typeof(entry(p, 1, first(indices(p))))))
    for (l, I) in enumerate(indices(p))
      s += f[l] * entry(p, j, I)
    end
    fHat[j] = s
  end
  fHat
end

# the adjoint of a complex plan, the transpose of a real one
function adjoint_direct!(f, p, fHat)
  check_size(f, size_in(p), "output"); check_size(fHat, size_out(p), "input")
  for (l, I) in enumerate(indices(p))
    s = zero(promote_type(eltype(fHat), typeof(entry(p, 1, I))))
    for j in 1:p.J
      s += fHat[j] * conj(entry(p, j, I))
    end
    f[l] = s
  end
  f
end

LinearAlgebra.mul!(f::AbstractArray, pl::Adjoint{Complex{T},<:Union{NDFTPlan{T},NNDFTPlan{T}}}, fHat::AbstractArray) where {T} =
  adjoint_direct!(f, parent(pl), fHat)
LinearAlgebra.mul!(f::AbstractArray, pl::Transpose{T,<:Union{NDCTPlan{T},NDSTPlan{T}}}, fHat::AbstractArray) where {T} =
  adjoint_direct!(f, parent(pl), fHat)
