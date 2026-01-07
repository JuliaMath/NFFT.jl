#=
using AbstractNFFTs: AbstractNFFTPlan, convolve!, convolve_transpose! # someday
using NFFT: NFFTPlan, convolve!, convolve_transpose! # currently
using LinearAlgebra: mul!
=#

#=
The following 2-step initialization helper function
is needed to accommodate GPU array types.
The more obvious statement `weights = fill(v, dims)`
can lead to the wrong array type and cause GPU tests to fail.
=#
function _fill_similar(array::AbstractArray, v::T, dims::Union{Integer,Dims}) where T
    weights = similar(array, T, dims)
    fill!(weights, v)
    return weights
end

#=
It would be desirable to use the memory pointed to by p.tmpVec
as working buffer for sdc iterations,
but this attempt led to intermittent corrupted outputs and errors.
=#
function _reinterpret_real(g::StridedArray{Complex{T}}) where {T <: Real}
    r1 = reinterpret(T, g)
    r2 = @view r1[1:2:end,:]
    return r2
end

#=
This method _almost_ conforms to the AbstractNFFT interface
except that it uses p.Ñ and p.tmpVec that are not part of that interface.
=#

"""
   weights = sdc(plan::NFFTPlan; iters=20, ...)

Compute weights for sample density compensation for given NFFT `plan`
Uses method of Pipe & Menon, Mag Reson Med, 44(1):179-186, Jan. 1999.
DOI: 10.1002/(SICI)1522-2594(199901)41:1<179::AID-MRM25>3.0.CO;2-V
The function applies `iters` iterations of that method.

The scaling here such that if the plan were 1D with N nodes equispaced
from -0.5 to 0.5-1/N, then the returned weights are ≈ 1/N.

The weights are scaled such that ``A' diag(w) A 1_N ≈ 1_N``.

The returned vector is real, positive values of length `plan.J`.

There are several named keyword arguments that are work buffers
that are all mutated: `weights, workg weights_tmp workf workv`.
If the caller provides all of those,
then this function should make only small allocations.
"""
function sdc(
    p::NFFTPlan{T,D,1};
    iters::Int = 20,
    # the following are working buffers that are all mutated:
    weights::AbstractVector{T} = _fill_similar(p.tmpVec, one(T), only(size_in(p))),
    weights_tmp::AbstractVector = similar(weights),
#   workg::AbstractArray = _reinterpret_real(p.tmpVec), # todo
    workg::AbstractArray = similar(p.tmpVec, T, p.Ñ),
    workf::AbstractVector = similar(p.tmpVec, Complex{T}, only(size_in(p))),
    workv::AbstractArray = similar(p.tmpVec, Complex{T}, size_out(p)),
) where {T <: Real, D}

  return sdc!(
    p,
    iters,
    weights,
    weights_tmp,
    workg,
    workf,
    workv,
  )
end


# ideally this function should be non-allocating
function sdc!(
    p::NFFTPlan{T,D,1},
    iters::Int,
    # the following are working buffers that are all mutated:
    weights::AbstractVector{T},
    weights_tmp::AbstractVector,
    workg::AbstractArray,
    workf::AbstractVector,
    workv::AbstractArray,
) where {T <: Real, D}

    scaling_factor = missing # will be set below

    # Pre-weighting to correct non-uniform sample density
    for i in 1:iters
        convolve_transpose!(p, weights, workg)
        if i == 1
            scaling_factor = maximum(workg)
        end

        workg ./= scaling_factor
        convolve!(p, workg, weights_tmp)
        weights_tmp ./= scaling_factor
        any(≤(0), weights_tmp) && throw("non-positive weights")
#       weights_tmp .+= eps(T) # todo: unnecessary?
        weights ./= weights_tmp
    end

    #=
    We want to scale the weights such that A' D(w) A 1_N ≈ 1_N.
    We find c ∈ ℝ that minimizes ‖u - c * v‖₂ where u ≜ 1_N
    and v ≜ A' D(w) A 1_N, and then scale w by that c.
    The analytical solution is c = real(u'v) / ‖v‖² = real(sum(v)) / ‖v‖².
    =#

#=
todo: can we cut this?
    # conversion to Array is a workaround for CuNFFT. Without it we get strange
    # results that indicate some synchronization issue
    f = Array( p * u )
    b = f .* Array(weights) # apply weights from above
    v = Array( adjoint(p) * convert(typeof(weights), b) )
    c = vec(v) \ vec(Array(u))  # least squares diff
    return abs.(convert(typeof(weights), c * Array(weights)))
=#

    # non converting version
    u = workv # trick to save memory
    fill!(u, one(T))
    mul!(workf, p, u)
    workf .*= weights # apply weights from above
    mul!(workv, adjoint(p), workf)
    c = real(sum(workv)) / sum(abs2, workv)

    weights .*= c
    return weights
end
