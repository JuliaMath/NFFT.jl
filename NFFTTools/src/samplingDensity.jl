#=
using AbstractNFFTs: AbstractNFFTPlan, convolve!, convolve_transpose!
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


"""
   weights = sdc(plan::AbstractNFFTPlan; iters=20)

Compute weights for sample density compensation for given NFFT `plan`
Uses method of Pipe & Menon, Mag Reson Med, 44(1):179-186, Jan. 1999.
DOI: 10.1002/(SICI)1522-2594(199901)41:1<179::AID-MRM25>3.0.CO;2-V

The scaling here such that if the plan were 1D with N nodes equispaced
from -0.5 to 0.5-1/N, then the returned weights are ≈ 1/N.

The returned vector is real, positive values of length `plan.J`.
"""
function sdc(p::AbstractNFFTPlan{T,D,1}; iters=20) where {T,D}
    #=
    We must initialize the weights to be a Complex vector of ones
    because the `convolve!` methods require Complex arrays.
    Nevertheless, the imaginary part of the weights
    will be zero throughout this entire algorithm
    because the interpolation kernel used in `convolve!` is real.
    In fact the weights will remain strictly positive.
    =#
    weights = _fill_similar(p.tmpVec, one(Complex{T}), p.J)
    weights_tmp = similar(weights)
    scaling_factor = missing # will be set below

    # Pre-weighting to correct non-uniform sample density
    for i in 1:iters
        convolve_transpose!(p, weights, p.tmpVec)
        @assert all(x -> imag(x) == 0, p.tmpVec) # should be real!
        if i == 1
            scaling_factor = maximum(real, p.tmpVec)
        end

        p.tmpVec ./= scaling_factor
        convolve!(p, p.tmpVec, weights_tmp)
        @assert all(z -> imag(z) == 0, weights_tmp) # should be real!
        weights_tmp ./= scaling_factor
        any(z -> real(z) ≤ 0, weights_tmp) && throw("non-positive weights")
        @. weights /= (weights_tmp + eps(T))
    end

    # Post weights to correct image scaling
    # This finds c, where ‖u - c * v‖₂² = 0 and then uses
    # c to scale all weights by a scalar factor.

    u = _fill_similar(weights, one(Complex{T}), p.N)

#=
    # conversion to Array is a workaround for CuNFFT. Without it we get strange
    # results that indicate some synchronization issue
    f = Array( p * u )
    b = f .* Array(weights) # apply weights from above
    v = Array( adjoint(p) * convert(typeof(weights), b) )
    c = vec(v) \ vec(Array(u))  # least squares diff
    return abs.(convert(typeof(weights), c * Array(weights)))
=#

    # non converting version
    f = similar(p.tmpVec, Complex{T}, p.J)
    mul!(f, p, u)
    f .*= weights # apply weights from above
    v = similar(p.tmpVec, Complex{T}, p.N)
    mul!(v, adjoint(p), f)
    c = vec(v) \ vec(u)  # least squares diff
    @assert c ≈ real(c) # should be real!
    c = real(c)
# todo: solve directly for real c after checking the math

#   return abs.(c * weights)
    return c * real(weights)
end
