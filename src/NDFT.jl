
### ndft functions ###

function ndft(plan::NFFTPlan{D}, f::AbstractArray{T,D}) where {D,T}
    plan.N == size(f) || throw(DimensionMismatch("Data is not consistent with NFFTPlan"))

    g = zeros(T, plan.M)

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

function ndft(x, f::AbstractArray{T,D}, rest...; kwargs...) where {D,T}
    p = NFFTPlan(x, size(f), rest...; kwargs...)
    return ndft(p, f)
end

function ndft_adjoint(plan::NFFTPlan{D}, fHat::AbstractArray{T,1}) where {D,T}
    plan.M == length(fHat) || throw(DimensionMismatch("Data is not consistent with NFFTPlan"))

    g = zeros(T, plan.N)

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

function ndft_adjoint(x, N, fHat::AbstractVector{T}, rest...; kwargs...) where T
    p = NFFTPlan(x, N, rest...; kwargs...)
    return ndft_adjoint(p, fHat)
end
