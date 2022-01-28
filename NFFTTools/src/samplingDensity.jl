
function sdc(p::AbstractNFFTPlan{T,D,1}; iters=20) where {T,D}
    # Weights for sample density compensation.
    # Uses method of Pipe & Menon, 1999. Mag Reson Med, 186, 179.
    weights = ones(Complex{T}, p.M)
    weights_tmp = similar(weights)
    scaling_factor = !isempty(p.B) ? maximum(p.B)^2 :  maximum(maximum(p.windowLUT))^2
    # Pre-weighting to correct non-uniform sample density
    for i in 1:iters
        convolve_adjoint!(p, weights, p.tmpVec)
        p.tmpVec ./= scaling_factor
        convolve!(p, p.tmpVec, weights_tmp)
        weights_tmp ./= scaling_factor
        for j in 1:length(weights)
            weights[j] = weights[j] / (abs(weights_tmp[j]) + eps(T))
        end
    end
    # Post weights to correct image scaling
    # This finds c, where ||u - c*v||_2^2 = 0 and then uses
    # c to scale all weights by a scalar factor.
    u = ones(Complex{T}, p.N)
    f = p * u
    f = f .* weights # apply weights from above
    v = adjoint(p) * f
    c = vec(v) \ vec(u)  # least squares diff
    abs.(c * weights) 
end
