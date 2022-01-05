
mutable struct NDFTPlan{T,D} <: AbstractNFFTPlan{T,D,1}
  N::NTuple{D,Int64}
  M::Int64
  x::Matrix{T}
end



function NDFTPlan(x::Matrix{T}, N::NTuple{D,Int}; kwargs...) where {T,D}

  if D != size(x,1)
    throw(ArgumentError("Nodes x have dimension $(size(x,1)) != $D"))
  end

  M = size(x, 2)

  return NDFTPlan{T,D}(N, M, x)
end


### ndft functions ###

ndft!(plan::NDFTPlan{Tp,D}, g::AbstractArray{Tg}, f::AbstractArray{T,D}) where {D,Tp,T,Tg} =
   nfft!(plan, g, f) where {D,T,Tg}

ndft_adjoint!(plan::NDFTPlan{Tp,D}, g::AbstractArray{Tg,D}, fHat::AbstractVector{T}) where {D,Tp,T,Tg} =
   nfft_adjoint!(plan, g, fHat) where {D,T,Tg}

ndft(plan::NDFTPlan{Tp,D}, f::AbstractArray{T,D}) where {Tp,T,D} =
   nfft!(plan, similar(f,plan.M), f)

ndft(x::AbstractArray, f::AbstractArray, rest...; kwargs...) =
   ndft(NDFTPlan(x, size(f), rest...; kwargs...), f)

ndft_adjoint(plan::NDFTPlan, fHat::AbstractVector) =
   nfft_adjoint!(plan, similar(fHat, plan.N), fHat)

ndft_adjoint(x, N, fHat::AbstractVector, rest...; kwargs...) =
   ndft_adjoint(NDFTPlan(x, N, rest...; kwargs...), fHat)



function AbstractNFFTs.nfft!(plan::NDFTPlan{Tp,D}, g::AbstractArray{Tg}, f::AbstractArray{T,D}) where {D,Tp,T,Tg}

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


function AbstractNFFTs.nfft_adjoint!(plan::NDFTPlan{Tp,D}, g::AbstractArray{Tg,D}, fHat::AbstractVector{T}) where {D,Tp,T,Tg}

    plan.M == length(fHat) ||
        throw(DimensionMismatch("Data f inconsistent with NDFTPlan"))
    plan.N == size(g) ||
        throw(DimensionMismatch("Output g inconsistent with NDFTPlan"))

    g .= zero(Tg)

    for l=1:prod(plan.N)
        idx = CartesianIndices(plan.N)[l]

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

