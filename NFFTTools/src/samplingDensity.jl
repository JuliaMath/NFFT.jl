
function sdc(p::AbstractNFFTPlan{T,D,1}; iters=20) where {T,D}
    # Weights for sample density compensation.
    # Uses method of Pipe & Menon, 1999. Mag Reson Med, 186, 179.
    weights = similar(p.tmpVec, Complex{T}, p.J)
    weights .= one(Complex{T})
    weights_tmp = similar(weights)
    scaling_factor = zero(T)

    # Pre-weighting to correct non-uniform sample density
    for i in 1:iters
        convolve_transpose!(p, weights, p.tmpVec)
        if i==1
         scaling_factor = maximum(abs.(p.tmpVec))
        end

        p.tmpVec ./= scaling_factor
        convolve!(p, p.tmpVec, weights_tmp)
        weights_tmp ./= scaling_factor
        weights ./= (abs.(weights_tmp) .+ eps(T))
    end
    # Post weights to correct image scaling
    # This finds c, where ||u - c*v||_2^2 = 0 and then uses
    # c to scale all weights by a scalar factor.
    u = similar(weights, Complex{T}, p.N) 
    u .= one(Complex{T})
    # conversion to Array is a workaround for CuNFFT. Without it we get strange
    # results that indicate some synchronization issue
    f = Array( p * u ) 
    b = f .* Array(weights) # apply weights from above
    v = Array( adjoint(p) * convert(typeof(weights), b) )
    c = vec(v) \ vec(Array(u))  # least squares diff
    
    return abs.(c * Array(weights)) 
end
