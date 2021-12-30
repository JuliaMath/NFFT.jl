
### ndft functions ###

"""
    ndft!(plan::NFFTPlan{D}, g::AbstractVector{Tg}, f::AbstractArray{T,D})

Compute NDFT of input array `f`
and store result in pre-allocated output array `g`.
Both arrays must have the same size compatible with the NFFT `plan`.

"""
function ndft!(plan::NFFTPlan{D}, g::AbstractArray{Tg}, f::AbstractArray{T,D}) where {D,T,Tg}

    plan.N == size(f) ||
        throw(DimensionMismatch("Data f is not consistent with NFFTPlan"))
    plan.M == length(g) ||
        throw(DimensionMismatch("Output g is inconsistent with NFFTPlan"))

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


"""
    ndft(plan::NFFTPlan{D}, f::AbstractArray{T,D})
    ndft(x, f::AbstractArray, rest...; kwargs...)

Non pre-allocated versions of NDFT; see `ndft!`.
"""
ndft(plan::NFFTPlan{D}, f::AbstractArray{T,D}) where {D,T} =
    ndft!(plan, similar(f,plan.M), f)

ndft(x, f::AbstractArray, rest...; kwargs...) =
    ndft(NFFTPlan(x, size(f), rest...; kwargs...), f)



"""
    ndft_adjoint!(plan::NFFTPlan{D}, g::AbstractArray{Tg,D}, fHat::AbstractVector)

Compute adjoint NDFT of input vector `fHat`
and store result in pre-allocated output array `g`.
The input arrays must have sizes compatible with the NFFT `plan`.
"""
function ndft_adjoint!(plan::NFFTPlan{D}, g::AbstractArray{Tg,D}, fHat::AbstractVector{T}) where {D,T,Tg}

    plan.M == length(fHat) ||
        throw(DimensionMismatch("Data f inconsistent with NFFTPlan"))
    plan.N == size(g) ||
        throw(DimensionMismatch("Output g inconsistent with NFFTPlan"))

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


"""
    ndft_adjoint(plan::NFFTPlan, fHat::AbstractVector)
    ndft_adjoint(x, N, fHat::AbstractVector, rest...; kwargs...)

Non pre-allocated versions of NDFT adjoint; see `ndft_adjoint!`.
"""
ndft_adjoint(plan::NFFTPlan, fHat::AbstractVector) =
    ndft_adjoint!(plan, similar(fHat, plan.N), fHat)

ndft_adjoint(x, N, fHat::AbstractVector, rest...; kwargs...) =
    ndft_adjoint(NFFTPlan(x, N, rest...; kwargs...), fHat)
